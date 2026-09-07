// Generic host adapter. The Python boundary projects a canonical assessed Schedule.
import Foundation
import Metal

struct Refusal: Error, CustomStringConvertible {
    let description: String
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Refusal(description: message) }
}

struct BufferSpec: Decodable {
    let name: String
    let dtype: String
    let shape: [Int]
    let size_bytes: Int
    let mode: String
    let input_path: String?
    let output_path: String
}

struct Manifest: Decodable {
    let target: String
    let source_language: String
    let compiler: String
    let language_standard: String
    let fast_math_enabled: Bool
    let threadgroup_memory_bytes: Int
    let execution_model: String
    let active_threads_per_threadgroup: Int
    let buffer_order: [String]
    let threads_per_threadgroup: [Int]
    let threadgroups_per_grid: [Int]
    let device_names: [String]
    let source_path: String
    let entry_point: String
    let buffers: [BufferSpec]
}

func size(_ values: [Int], _ name: String) throws -> MTLSize {
    try require(values.count == 3 && values.allSatisfy { $0 > 0 && $0 <= Int(UInt32.max) },
                "\(name) requires three positive uint32 dimensions")
    return MTLSize(width: values[0], height: values[1], depth: values[2])
}


func loadManifest(_ manifestPath: String) throws -> (Manifest, [Data?]) {
    try require(manifestPath.hasPrefix("/"), "manifest path must be absolute")
    let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
    let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    let fields: Set<String> = ["target", "source_language", "compiler", "language_standard",
        "fast_math_enabled", "threadgroup_memory_bytes", "execution_model",
        "active_threads_per_threadgroup", "buffer_order", "threads_per_threadgroup",
        "threadgroups_per_grid", "device_names", "source_path", "entry_point", "buffers"]
    try require(raw != nil && Set(raw!.keys) == fields, "manifest fields differ")
    let bufferFields: Set<String> = ["name", "dtype", "shape", "size_bytes", "mode", "input_path", "output_path"]
    guard let rawBuffers = raw!["buffers"] as? [[String: Any]] else {
        throw Refusal(description: "manifest buffers must be objects")
    }
    try require(rawBuffers.allSatisfy { Set($0.keys) == bufferFields }, "buffer fields differ")
    let manifest = try JSONDecoder().decode(Manifest.self, from: data)
    try require(["apple_gpu_family7", "apple_gpu_family8"].contains(manifest.target) && manifest.source_language == "metal" &&
                manifest.compiler == "MTLDevice.makeLibrary", "unsupported exact Metal route")
    try require(manifest.language_standard == "metal2.3" && !manifest.fast_math_enabled &&
                manifest.threadgroup_memory_bytes == 0 &&
                ((manifest.execution_model == "serial_program_tile" && manifest.active_threads_per_threadgroup == 1) ||
                 (manifest.execution_model == "simd_program_tile" && manifest.active_threads_per_threadgroup == 32)),
                "unsupported Metal language, math, memory or execution commitment")
    _ = try size(manifest.threadgroups_per_grid, "threadgroups_per_grid")
    _ = try size(manifest.threads_per_threadgroup, "threads_per_threadgroup")
    try require(manifest.threads_per_threadgroup == [32, 1, 1], "unsupported role mapping")
    try require(!manifest.device_names.isEmpty && !manifest.device_names.contains(""),
                "exact device names are required")
    try require(manifest.source_path.hasPrefix("/") && !manifest.entry_point.isEmpty,
                "absolute source path and entry point required")
    try require(!manifest.buffers.isEmpty && manifest.buffers.count <= 31 &&
                manifest.buffer_order == manifest.buffers.map { $0.name } &&
                Set(manifest.buffer_order).count == manifest.buffer_order.count,
                "buffer bindings differ")
    var inputData: [Data?] = []
    var outputPaths = Set<String>()
    for buffer in manifest.buffers {
        try require(buffer.dtype == "fp32" && ["input", "output"].contains(buffer.mode) &&
                    !buffer.shape.isEmpty && buffer.shape.allSatisfy { $0 > 0 },
                    "unsupported buffer declaration: \(buffer.name)")
        var byteCount = 4
        for extent in buffer.shape {
            let product = byteCount.multipliedReportingOverflow(by: extent)
            try require(!product.overflow && product.partialValue <= Int(Int32.max),
                        "buffer shape exceeds supported size: \(buffer.name)")
            byteCount = product.partialValue
        }
        try require(byteCount == buffer.size_bytes, "buffer shape/byte size differs: \(buffer.name)")
        try require(buffer.output_path.hasPrefix("/") &&
                    !FileManager.default.fileExists(atPath: buffer.output_path) &&
                    outputPaths.insert(buffer.output_path).inserted,
                    "buffer output must be a fresh absolute path: \(buffer.name)")
        if buffer.mode == "input" {
            try require(buffer.input_path?.hasPrefix("/") == true,
                        "input path missing: \(buffer.name)")
            let bytes = try Data(contentsOf: URL(fileURLWithPath: buffer.input_path!))
            try require(bytes.count == byteCount, "input byte length differs: \(buffer.name)")
            let finite = bytes.withUnsafeBytes { raw in
                stride(from: 0, to: raw.count, by: 4).allSatisfy {
                    Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self))).isFinite
                }
            }
            try require(finite, "finite FP32 input required: \(buffer.name)")
            inputData.append(bytes)
        } else {
            try require(buffer.input_path == nil, "output buffer cannot supply input bytes")
            inputData.append(nil)
        }
    }
    return (manifest, inputData)
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

final class Prepared {
    let manifest: Manifest
    let inputData: [Data?]
    let pipeline: MTLComputePipelineState
    var buffers: [MTLBuffer] = []
    let grid: MTLSize
    let threads: MTLSize
    let coldPrepareSeconds: Double
    let coldLibraryPipelineSeconds: Double

    init(path: String, device: MTLDevice) throws {
        let begin = now()
        (manifest, inputData) = try loadManifest(path)
        grid = try size(manifest.threadgroups_per_grid, "threadgroups_per_grid")
        threads = try size(manifest.threads_per_threadgroup, "threads_per_threadgroup")
    let exactDevice = manifest.target == "apple_gpu_family7" ? "Apple M1 Pro" : "Apple M2"
    let exactFamily = manifest.target == "apple_gpu_family7"
        ? device.supportsFamily(.apple7) && !device.supportsFamily(.apple8)
        : device.supportsFamily(.apple8) && !device.supportsFamily(.apple9)
    try require(manifest.device_names == [exactDevice] && device.name == exactDevice && exactFamily,
                "exact target/device mismatch: observed \(device.name)")
    try require(device.hasUnifiedMemory, "shared host buffers require unified memory")
    let maxThreads = device.maxThreadsPerThreadgroup
    try require(threads.width <= maxThreads.width && threads.height <= maxThreads.height &&
                threads.depth <= maxThreads.depth, "thread dimensions exceed device limits")
    for buffer in manifest.buffers {
        try require(buffer.size_bytes <= device.maxBufferLength,
                    "buffer exceeds device limit: \(buffer.name)")
    }
    let source = try String(contentsOfFile: manifest.source_path, encoding: .utf8)
    let options = MTLCompileOptions()
    options.languageVersion = .version2_3
    options.mathMode = .safe
    options.mathFloatingPointFunctions = .precise
    let libraryStart = now()
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: manifest.entry_point) else {
        throw Refusal(description: "emitted entry point unavailable")
    }
    pipeline = try device.makeComputePipelineState(function: function)
    try require(pipeline.maxTotalThreadsPerThreadgroup >= 32 &&
                pipeline.threadExecutionWidth == 32 &&
                pipeline.staticThreadgroupMemoryLength == 0,
                "compiled pipeline violates thread/memory commitments")
    coldLibraryPipelineSeconds = now() - libraryStart

        for (index, spec) in manifest.buffers.enumerated() {
            guard let buffer = device.makeBuffer(length: spec.size_bytes, options: .storageModeShared) else {
                throw Refusal(description: "buffer allocation failed: \(spec.name)")
            }
            if let input = inputData[index] {
                input.withUnsafeBytes { bytes in
                    buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: input.count)
                }
            } else {
                // A missing output store cannot accidentally compare equal to zero.
                buffer.contents().bindMemory(to: UInt32.self, capacity: spec.size_bytes / 4)
                    .initialize(repeating: Float.nan.bitPattern, count: spec.size_bytes / 4)
            }
            buffers.append(buffer)
        }
        coldPrepareSeconds = now() - begin
    }
}

extension Prepared {
    func poisonOutputs() {
        for (buffer, spec) in zip(buffers, manifest.buffers) where spec.mode == "output" {
            buffer.contents().bindMemory(to: UInt32.self, capacity: spec.size_bytes / 4)
                .update(repeating: Float.nan.bitPattern, count: spec.size_bytes / 4)
        }
    }

    func dispatch(queue: MTLCommandQueue, count: Int,
                  counter: MTLCounterSampleBuffer? = nil) throws -> [String: Any] {
        poisonOutputs() // excluded from warmed synchronous host-call time
        let begin = now()
        guard let command = queue.makeCommandBuffer() else {
            throw Refusal(description: "Metal command creation failed")
        }
        let descriptor = MTLComputePassDescriptor()
        descriptor.dispatchType = .serial
        if let counter = counter {
            descriptor.sampleBufferAttachments[0].sampleBuffer = counter
            descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0
            descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1
        }
        guard let encoder = command.makeComputeCommandEncoder(descriptor: descriptor) else {
            throw Refusal(description: "Metal encoder creation failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        for _ in 0..<count { encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads) }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        let hostSeconds = now() - begin
        try require(command.status == .completed && command.error == nil,
                    "Metal command failed: \(String(describing: command.error))")
        let gpuSeconds = command.gpuEndTime - command.gpuStartTime
        return ["command_status": "completed", "dispatches": count,
                "warmed_host_call_seconds": hostSeconds,
                "gpu_start_time_seconds": command.gpuStartTime,
                "gpu_end_time_seconds": command.gpuEndTime,
                "gpu_command_buffer_seconds": gpuSeconds,
                "amortized_dispatch_seconds": gpuSeconds / Double(count)]
    }

    func validate(oraclePath: String) throws -> [String: Any] {
        struct Oracle: Decodable { let expected: [Double]; let absolute_tolerance: [Double] }
        let oracles = try JSONDecoder().decode([String: Oracle].self,
            from: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
        try require(Set(oracles.keys) == Set(manifest.buffers.filter { $0.mode == "output" }.map { $0.name }),
                    "oracle output names differ from manifest")
        var maxError = 0.0
        for (index, spec) in manifest.buffers.enumerated() {
            let bytes = Data(bytes: buffers[index].contents(), count: spec.size_bytes)
            if let input = inputData[index] {
                try require(bytes == input, "input mutation: \(spec.name)")
            } else {
                let oracle = oracles[spec.name]!
                try require(oracle.expected.count * 4 == spec.size_bytes &&
                            oracle.absolute_tolerance.count == oracle.expected.count,
                            "oracle size differs: \(spec.name)")
                for i in oracle.expected.indices {
                    let expected = oracle.expected[i]
                    let bound = oracle.absolute_tolerance[i]
                    let actual = Double(buffers[index].contents().load(fromByteOffset: i * 4, as: Float.self))
                    let error = abs(actual - expected)
                    try require(expected.isFinite && bound.isFinite && bound >= 0 && actual.isFinite && error <= bound,
                                "output \(spec.name)[\(i)]=\(actual), reference=\(expected), tolerance=\(bound)")
                    maxError = max(maxError, error)
                }
            }
        }
        return ["gpu_correctness": "passed", "input_immutability": "passed", "max_abs_error": maxError]
    }

    func writeOutputs() throws {
        for (buffer, spec) in zip(buffers, manifest.buffers) {
            try Data(bytes: buffer.contents(), count: spec.size_bytes)
                .write(to: URL(fileURLWithPath: spec.output_path), options: .withoutOverwriting)
        }
    }
}

func capabilities(_ device: MTLDevice) -> [String: Any] {
    ["device": device.name, "counter_sets": (device.counterSets ?? []).map {
        ["name": $0.name, "counters": $0.counters.map { $0.name }]
    }, "dispatch_boundary": device.supportsCounterSampling(.atDispatchBoundary),
       "stage_boundary": device.supportsCounterSampling(.atStageBoundary)]
}

func profile(_ prepared: Prepared, device: MTLDevice, queue: MTLCommandQueue) throws -> [String: Any] {
    guard device.supportsCounterSampling(.atStageBoundary),
          let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
        return ["coverage": "unavailable", "reason": "timestamp counter set or compute-stage boundary unsupported"]
    }
    let descriptor = MTLCounterSampleBufferDescriptor()
    descriptor.counterSet = set
    descriptor.storageMode = .shared
    descriptor.sampleCount = 2
    let counter = try device.makeCounterSampleBuffer(descriptor: descriptor)
    let observed = try prepared.dispatch(queue: queue, count: 1, counter: counter)
    try validTimer(observed)
    guard let data = try counter.resolveCounterRange(0..<2), data.count == 16 else {
        throw Refusal(description: "timestamp counter resolution failed")
    }
    let ticks = data.withUnsafeBytes { raw in
        [raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self), raw.loadUnaligned(fromByteOffset: 8, as: UInt64.self)]
    }
    try require(ticks[0] > 0 && ticks[1] > ticks[0] && ticks[1] != UInt64.max,
                "invalid instrumented timestamp samples")
    return ["coverage": "compute_stage_timestamp", "raw_counter_timestamps": ticks,
            "counter_set": set.name, "requested_counter": "GPUTimestamp", "sampling_boundary": "compute_stage",
            "counter_units": "raw device timestamp units, no clock conversion asserted",
            "instrumented_command": observed,
            "dispatch_boundary_sampling": device.supportsCounterSampling(.atDispatchBoundary) ? "available_not_requested" : "unsupported",
            "not_collected": ["occupancy", "bandwidth", "instruction_counters"]]
}

struct Artifact: Decodable {
    let id: String
    let manifest_path: String
    let oracle_path: String
    let origin: String
}
struct Batch: Decodable {
    let artifacts: [Artifact]
    let warmups: Int
    let pilot_samples: Int
    let max_batch_dispatches: Int
    let target_command_seconds: Double
    let orders: [[[String]]]
}

func median(_ values: [Double]) -> Double {
    let ordered = values.sorted(); let middle = ordered.count / 2
    return ordered.count % 2 == 0 ? (ordered[middle - 1] + ordered[middle]) / 2 : ordered[middle]
}

func validTimer(_ sample: [String: Any]) throws {
    let gpu = sample["gpu_command_buffer_seconds"] as? Double ?? 0
    let start = sample["gpu_start_time_seconds"] as? Double ?? 0
    let host = sample["warmed_host_call_seconds"] as? Double ?? 0
    try require(gpu.isFinite && gpu > 0 && start.isFinite && start > 0 && host.isFinite && host > 0,
                "invalid GPU command-buffer or host timer")
}

func executeBatch(path: String, device: MTLDevice, queue: MTLCommandQueue,
                  compileOnly: Bool) throws -> [String: Any] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    let fields: Set<String> = ["artifacts", "warmups", "pilot_samples", "max_batch_dispatches", "target_command_seconds", "orders"]
    try require(raw != nil && Set(raw!.keys) == fields, "unsupported batch field")
    guard let rawArtifacts = raw!["artifacts"] as? [[String: Any]] else {
        throw Refusal(description: "batch artifacts must be objects")
    }
    let artifactFields: Set<String> = ["id", "manifest_path", "oracle_path", "origin"]
    try require(rawArtifacts.allSatisfy { Set($0.keys) == artifactFields }, "unsupported artifact field")
    let batch = try JSONDecoder().decode(Batch.self, from: data)
    let ids = Set(batch.artifacts.map { $0.id })
    try require(batch.artifacts.count >= 2 && batch.artifacts.count <= 9 && ids.count == batch.artifacts.count &&
                batch.artifacts.filter { $0.origin == "compiler_generated" }.count <= 6 &&
                batch.artifacts.allSatisfy { ["compiler_generated", "handwritten_reference", "released_replay"].contains($0.origin) &&
                    $0.manifest_path.hasPrefix("/") && $0.oracle_path.hasPrefix("/") }, "invalid batch artifacts/provenance")
    try require((1...5).contains(batch.warmups) && (1...5).contains(batch.pilot_samples) &&
                (1...128).contains(batch.max_batch_dispatches) &&
                batch.target_command_seconds.isFinite && batch.target_command_seconds > 0 && batch.target_command_seconds <= 0.01 &&
                (1...3).contains(batch.orders.count), "measurement budget differs")
    let candidateIDs = Set(batch.artifacts.filter { $0.origin == "compiler_generated" }.map { $0.id })
    let referenceIDs = ids.subtracting(candidateIDs)
    let confirmationIDs = referenceIDs.union(["__selected__"])
    try require(!candidateIDs.isEmpty && referenceIDs.count >= 2 && !ids.contains("__selected__") && batch.orders.count == 3,
                "one search and two independent confirmations require candidates")
    for (round, sweeps) in batch.orders.enumerated() {
        let expected = round == 0 ? ids : confirmationIDs
        try require(!sweeps.isEmpty && sweeps.count <= 12 && sweeps.allSatisfy { Set($0) == expected && $0.count == expected.count },
                    "matched sweep treatment slots differ")
    }
    let events = URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("events.jsonl")
    try Data().write(to: events, options: .withoutOverwriting)
    let handle = try FileHandle(forWritingTo: events)
    defer { try? handle.close() }
    func event(_ item: [String: Any]) throws {
        try handle.write(contentsOf: JSONSerialization.data(withJSONObject: item, options: [.sortedKeys]))
        try handle.write(contentsOf: Data("\n".utf8))
        try handle.synchronize()
    }
    var cache: [String: Prepared] = [:]
    var prepared: [String: Prepared] = [:]
    var construction: [String: Any] = [:]
    for artifact in batch.artifacts {
        try event(["stage": "compile", "artifact": artifact.id, "status": "started"])
        let shared = cache[artifact.manifest_path] != nil
        let item = try cache[artifact.manifest_path] ?? Prepared(path: artifact.manifest_path, device: device)
        cache[artifact.manifest_path] = item
        prepared[artifact.id] = item
        construction[artifact.id] = ["cold_prepare_seconds": item.coldPrepareSeconds,
            "cold_library_pipeline_seconds": item.coldLibraryPipelineSeconds, "reused_prepared_object": shared,
            "execution_model": item.manifest.execution_model, "origin": artifact.origin]
        try event(["stage": "compile", "artifact": artifact.id, "status": "completed"])
    }
    if compileOnly { return ["status": "compiled", "construction": construction] }
    var validation: [String: Any] = [:]
    for artifact in batch.artifacts {
        let item = prepared[artifact.id]!
        try event(["stage": "correctness", "artifact": artifact.id, "status": "started"])
        _ = try item.dispatch(queue: queue, count: 1)
        validation[artifact.id] = try item.validate(oraclePath: artifact.oracle_path)
        try event(["stage": "correctness", "artifact": artifact.id, "status": "passed"])
        for _ in 0..<batch.warmups { _ = try item.dispatch(queue: queue, count: 1) }
    }
    var pilot: [[String: Any]] = []
    var fastest = Double.infinity
    for artifact in batch.artifacts where artifact.origin != "compiler_generated" {
        var armTimes: [Double] = []
        for i in 0..<batch.pilot_samples {
            try event(["stage": "pilot", "status": "started", "artifact": artifact.id, "sample": i])
            var sample = try prepared[artifact.id]!.dispatch(queue: queue, count: 1)
            try validTimer(sample)
            sample["validation"] = try prepared[artifact.id]!.validate(oraclePath: artifact.oracle_path)
            sample["artifact"] = artifact.id; sample["sample"] = i; sample["stage"] = "pilot"
            armTimes.append(sample["gpu_command_buffer_seconds"] as! Double)
            pilot.append(sample); try event(sample)
        }
        fastest = min(fastest, median(armTimes))
    }
    let powers = [1, 2, 4, 8, 16, 32, 64, 128].filter { $0 <= batch.max_batch_dispatches }
    let dispatches = powers.first { Double($0) * fastest >= batch.target_command_seconds } ?? powers.last!
    var selected: String?
    var samples: [[String: Any]] = []
    for (round, sweeps) in batch.orders.enumerated() {
        for (sweep, order) in sweeps.enumerated() {
            for (position, slot) in order.enumerated() {
                let id = slot == "__selected__" ? selected! : slot
                try event(["stage": "measurement", "status": "started", "artifact": id, "round": round, "sweep": sweep, "position": position])
                var sample = try prepared[id]!.dispatch(queue: queue, count: dispatches)
                try validTimer(sample)
                let artifact = batch.artifacts.first { $0.id == id }!
                sample["validation"] = try prepared[id]!.validate(oraclePath: artifact.oracle_path)
                sample["artifact"] = id; sample["round"] = round; sample["sweep"] = sweep
                sample["position"] = position; sample["stage"] = "measurement"
                samples.append(sample); try event(sample)
            }
        }
        if round == 0 {
            selected = candidateIDs.sorted().min { left, right in
                let leftTimes = samples.filter { $0["artifact"] as? String == left }.map { $0["gpu_command_buffer_seconds"] as! Double }
                let rightTimes = samples.filter { $0["artifact"] as? String == right }.map { $0["gpu_command_buffer_seconds"] as! Double }
                return median(leftTimes) < median(rightTimes)
            }
            try event(["stage": "selection", "artifact": selected!, "basis": "lowest search median GPU command-buffer time"])
        }
    }
    var profiles: [String: Any] = [:]
    for artifact in batch.artifacts {
        let item = prepared[artifact.id]!
        try event(["stage": "instrumented_observation", "artifact": artifact.id, "status": "started"])
        var observation = try profile(item, device: device, queue: queue)
        if observation["coverage"] as? String != "unavailable" {
            observation["validation"] = try item.validate(oraclePath: artifact.oracle_path)
        }
        profiles[artifact.id] = observation
        try event(["stage": "instrumented_observation", "artifact": artifact.id, "observation": profiles[artifact.id]!])
    }
    for item in cache.values { try item.writeOutputs() }
    return ["status": "completed", "command_status": "completed", "construction": construction,
            "correctness": validation, "pilot_samples": pilot, "raw_samples": samples,
            "batch_dispatches": dispatches, "instrumented_observations": profiles,
            "ordinary_samples_instrumented": false, "warmups_per_artifact": batch.warmups,
            "selected_candidate": selected!, "cache_policy": "warm buffer reuse, no cache flush",
            "dispatch_ordering": "MTLDispatchType.serial, one encoder and completion per batch"]
}

func execute() throws {
    let args = CommandLine.arguments
    try require((args.count == 2 && args[1] == "--inspect") ||
                (args.count == 3 && ["--compile-only", "--run", "--batch", "--compile-batch"].contains(args[1])),
                "usage: metal-runner --inspect | --run|--compile-only MANIFEST | --batch|--compile-batch BATCH")
    let begin = now()
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Refusal(description: "no Metal device/queue available")
    }
    let deviceQueueSeconds = now() - begin
    var receipt: [String: Any]
    if args[1] == "--inspect" {
        receipt = ["status": "inspected", "capabilities": capabilities(device)]
    } else if args[1] == "--batch" || args[1] == "--compile-batch" {
        try require(args[2].hasPrefix("/"), "batch path must be absolute")
        receipt = try executeBatch(path: args[2], device: device, queue: queue, compileOnly: args[1] == "--compile-batch")
    } else {
        let prepared = try Prepared(path: args[2], device: device)
        receipt = ["status": "compiled", "execution_model": prepared.manifest.execution_model,
                   "active_threads_per_threadgroup": prepared.manifest.active_threads_per_threadgroup,
                   "cold_prepare_seconds": prepared.coldPrepareSeconds,
                   "cold_library_pipeline_seconds": prepared.coldLibraryPipelineSeconds]
        if args[1] == "--run" {
            _ = try prepared.dispatch(queue: queue, count: 1)
            try prepared.writeOutputs()
            receipt["status"] = "completed"; receipt["command_status"] = "completed"
        }
    }
    receipt["device"] = device.name
    receipt["target"] = device.name == "Apple M1 Pro" ? "apple_gpu_family7"
        : device.name == "Apple M2" ? "apple_gpu_family8" : "unsupported"
    receipt["capabilities"] = capabilities(device)
    receipt["device_queue_construction_seconds"] = deviceQueueSeconds
    receipt["math_mode"] = "safe"; receipt["math_functions"] = "precise"; receipt["language_standard"] = "metal2.3"
    let result = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    FileHandle.standardOutput.write(result)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

do { try execute() } catch {
    FileHandle.standardError.write(Data("Metal runtime refusal: \(error)\n".utf8))
    exit(1)
}
