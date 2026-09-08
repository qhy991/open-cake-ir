// Closed observer for one existing common Evaluation request. No source compilation.
import Foundation
import Metal

struct Tensor: Decodable { let name: String; let shape: [Int]; let dtype: String; let mode: String }
struct Manifest: Decodable {
    let target: String; let kernel_name: String; let tensor_abi: [Tensor]; let grid: [Int]; let block: [Int]
}
struct Participant: Decodable { let role: String; let archive_path: String; let manifest: Manifest }
struct OraclePaths: Decodable { let expected: String; let tolerance: String }
struct InputCase: Decodable { let id: String; let inputs: [String: String]; let oracles: [String: OraclePaths] }
struct Launch: Decodable {
    let index: Int; let role: String; let phase: String; let input_case_id: String; let timed: Bool; let profile: Bool
    let pair_index: Int?; let position: Int?
}
struct Request: Decodable {
    let expected_host: [String: String]; let participants: [Participant]; let cases: [InputCase]
    let launches: [Launch]; let output_directory: String
}
struct Refusal: Error, CustomStringConvertible { let description: String }
func require(_ value: Bool, _ message: String) throws { if !value { throw Refusal(description: message) } }
// Only persistence moves: copies and validation remain immediate. The bound is on
// additional pending snapshot payload, not total process or Metal buffer memory.
let snapshotPayloadLimit = 64 * 1024 * 1024
struct SnapshotCohort: Hashable { let pair: Int; let position: Int; let role: String }
func snapshotFlushPlan(_ launches: [Launch], participants: [Participant], limit: Int) throws -> Set<Int> {
    var sizes: [String: Int] = [:]
    for participant in participants {
        var total = 0
        for tensor in participant.manifest.tensor_abi {
            var bytes = 4
            for extent in tensor.shape {
                let product = bytes.multipliedReportingOverflow(by: extent)
                try require(extent > 0 && !product.overflow, "snapshot tensor byte count overflow")
                bytes = product.partialValue
            }
            let sum = total.addingReportingOverflow(bytes)
            try require(!sum.overflow, "snapshot participant byte count overflow")
            total = sum.partialValue
        }
        try require(sizes[participant.role] == nil, "duplicate snapshot participant")
        sizes[participant.role] = total
    }
    var boundaries = Set<Int>(); var seen = Set<SnapshotCohort>()
    var active: SnapshotCohort? = nil; var activeCase = ""; var pending = 0
    for (index, launch) in launches.enumerated() {
        try require(launch.index == index && sizes[launch.role] != nil, "snapshot launch sequence or participant differs")
        var cohort: SnapshotCohort? = nil
        if launch.phase == "cohort" {
            guard let pair = launch.pair_index, let position = launch.position else { throw Refusal(description: "cohort snapshot requires existing pair and position") }
            try require(pair >= 0 && (position == 0 || position == 1) && !launch.profile, "cohort snapshot metadata differs")
            cohort = SnapshotCohort(pair: pair, position: position, role: launch.role)
        } else {
            try require(launch.pair_index == nil && launch.position == nil, "non-cohort snapshot has cohort coordinates")
        }
        if cohort != active {
            if active != nil { boundaries.insert(index - 1) }
            pending = 0; active = cohort; activeCase = launch.input_case_id
            if let cohort = cohort {
                try require(seen.insert(cohort).inserted, "snapshot cohort must remain contiguous")
            }
        }
        if cohort != nil {
            try require(activeCase == launch.input_case_id, "snapshot cohort input case differs")
            let bytes = sizes[launch.role]!
            try require(bytes <= limit && pending <= limit - bytes, "cohort snapshot payload exceeds 64 MiB bound before dispatch")
            pending += bytes
        } else { boundaries.insert(index) }
    }
    if active != nil { boundaries.insert(launches.count - 1) }
    return boundaries
}
final class OwnedSnapshots {
    let directory: URL; let limit: Int
    let write: (Data, URL) throws -> Void
    private var pending: [(Data, URL)] = []
    private(set) var pendingBytes = 0
    private(set) var peakPendingBytes = 0
    private(set) var failedWrites: [String] = []
    init(directory: URL, limit: Int = snapshotPayloadLimit,
         write: @escaping (Data, URL) throws -> Void = { try $0.write(to: $1, options: .withoutOverwriting) }) {
        self.directory = directory; self.limit = limit; self.write = write
    }
    func capture(_ pointer: UnsafeRawPointer, count: Int, name: String, deferred: Bool) throws -> Data {
        try require(count >= 0 && (!deferred || (count <= limit && pendingBytes <= limit - count)), "pending snapshot bound differs from preflight")
        let data = Data(bytes: pointer, count: count) // Owned immediately; never bytesNoCopy or an MTLBuffer reference.
        let path = directory.appendingPathComponent(name)
        if deferred {
            pending.append((data, path)); pendingBytes += count
            peakPendingBytes = max(peakPendingBytes, pendingBytes)
        } else {
            do { try write(data, path) } catch { failedWrites.append(path.lastPathComponent); throw error }
        }
        return data
    }
    func flush() throws {
        // Attempt each pending file once even if another write fails. Clearing the
        // queue first prevents the outer error handler from retrying failed writes.
        let batch = pending; pending = []; pendingBytes = 0
        var firstError: Error? = nil
        for (data, path) in batch {
            do { try write(data, path) }
            catch { failedWrites.append(path.lastPathComponent); if firstError == nil { firstError = error } }
        }
        if let firstError = firstError { throw firstError }
    }
}
func flushAfterFailure(_ snapshots: OwnedSnapshots?, original: Error) -> (Error, Error?) {
    do { try snapshots?.flush(); return (original, nil) }
    catch { return (original, error) }
}
func errorDetails(_ error: Error) -> [String: Any] {
    let native = error as NSError
    return ["domain": native.domain, "code": native.code,
            "description": (error as? Refusal)?.description ?? native.localizedDescription]
}

func doubles(_ path: String) throws -> [Double] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    try require(data.count % 8 == 0, "oracle Double byte length differs")
    return data.withUnsafeBytes { raw in (0..<(data.count / 8)).map { raw.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self) } }
}
final class Prepared {
    let manifest: Manifest
    let pipeline: MTLComputePipelineState
    let device: MTLDevice
    let inputs: [String: [String: MTLBuffer]]
    let inputBytes: [String: [String: Data]]
    let oracles: [String: [String: ([Double], [Double])]]
    init(_ participant: Participant, cases: [InputCase], device: MTLDevice) throws {
        self.device = device; manifest = participant.manifest
        try require(manifest.block == [32, 1, 1] && manifest.grid.count == 3 && manifest.grid.allSatisfy { $0 > 0 }, "launch geometry differs")
        let archiveURL = URL(fileURLWithPath: participant.archive_path)
        let descriptor = MTLBinaryArchiveDescriptor(); descriptor.url = archiveURL
        let archive = try device.makeBinaryArchive(descriptor: descriptor)
        let library = try device.makeLibrary(URL: archiveURL)
        guard let function = library.makeFunction(name: manifest.kernel_name) else { throw Refusal(description: "archive entry point unavailable") }
        let pipelineDescriptor = MTLComputePipelineDescriptor()
        pipelineDescriptor.computeFunction = function; pipelineDescriptor.binaryArchives = [archive]
        pipeline = try device.makeComputePipelineState(descriptor: pipelineDescriptor, options: [.failOnBinaryArchiveMiss], reflection: nil)
        try require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 32 && pipeline.staticThreadgroupMemoryLength == 0, "pipeline SIMD resources differ")
        var allInputs: [String: [String: MTLBuffer]] = [:]
        var allBytes: [String: [String: Data]] = [:]
        var allOracles: [String: [String: ([Double], [Double])]] = [:]
        for inputCase in cases {
            var buffers: [String: MTLBuffer] = [:]; var bytes: [String: Data] = [:]
            var oracle: [String: ([Double], [Double])] = [:]
            for tensor in manifest.tensor_abi {
                try require(tensor.dtype == "fp32" && tensor.shape.allSatisfy { $0 > 0 }, "only finite FP32 tensor ABI admitted")
                let length = tensor.shape.reduce(1, *) * 4
                try require(length <= device.maxBufferLength, "tensor exceeds device buffer limit")
                if tensor.mode == "input" {
                    guard let path = inputCase.inputs[tensor.name] else { throw Refusal(description: "input buffer missing") }
                    let data = try Data(contentsOf: URL(fileURLWithPath: path))
                    try require(data.count == length, "input buffer byte length differs")
                    let buffer = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: length, options: .storageModeShared) }
                    guard let buffer = buffer else { throw Refusal(description: "input allocation failed") }
                    buffers[tensor.name] = buffer; bytes[tensor.name] = data
                } else {
                    guard let paths = inputCase.oracles[tensor.name] else { throw Refusal(description: "output oracle missing") }
                    let expected = try doubles(paths.expected); let bounds = try doubles(paths.tolerance)
                    try require(expected.count * 4 == length && bounds.count == expected.count
                                && expected.allSatisfy { $0.isFinite } && bounds.allSatisfy { $0.isFinite && $0 >= 0 }, "oracle shape or numeric domain differs")
                    oracle[tensor.name] = (expected, bounds)
                }
            }
            allInputs[inputCase.id] = buffers; allBytes[inputCase.id] = bytes; allOracles[inputCase.id] = oracle
        }
        inputs = allInputs; inputBytes = allBytes; oracles = allOracles
    }
    func observe(_ launch: Launch, queue: MTLCommandQueue, snapshots: OwnedSnapshots) throws -> [String: Any] {
        guard let caseInputs = inputs[launch.input_case_id], let caseBytes = inputBytes[launch.input_case_id],
              let caseOracles = oracles[launch.input_case_id] else { throw Refusal(description: "unknown validation input case") }
        var buffers: [MTLBuffer] = []
        for tensor in manifest.tensor_abi {
            if let input = caseInputs[tensor.name] { buffers.append(input); continue }
            let count = tensor.shape.reduce(1, *)
            guard let output = device.makeBuffer(length: count * 4, options: .storageModeShared) else { throw Refusal(description: "fresh output allocation failed") }
            output.contents().bindMemory(to: Float.self, capacity: count).initialize(repeating: Float.nan, count: count)
            buffers.append(output)
        }
        let computePass = MTLComputePassDescriptor()
        var counter: MTLCounterSampleBuffer? = nil
        if launch.profile {
            guard device.supportsCounterSampling(.atStageBoundary),
                  let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }),
                  set.counters.contains(where: { $0.name == "GPUTimestamp" }) else { throw Refusal(description: "GPUTimestamp compute-stage sampling unavailable") }
            let descriptor = MTLCounterSampleBufferDescriptor()
            descriptor.counterSet = set; descriptor.storageMode = .shared; descriptor.sampleCount = 2
            counter = try device.makeCounterSampleBuffer(descriptor: descriptor)
            computePass.sampleBufferAttachments[0].sampleBuffer = counter
            computePass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0
            computePass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1
        }
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder(descriptor: computePass) else { throw Refusal(description: "Metal encoder creation failed") }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreadgroups(MTLSize(width: manifest.grid[0], height: manifest.grid[1], depth: manifest.grid[2]),
                                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed && command.error == nil, "Metal command did not complete")
        try require(command.gpuStartTime.isFinite && command.gpuEndTime.isFinite
                    && command.gpuStartTime > 0 && command.gpuEndTime > command.gpuStartTime, "invalid observed command timestamps")
        let commandObservation: [String: Any] = ["launch_index": launch.index, "completed": true,
            "gpu_start_seconds": command.gpuStartTime, "gpu_end_seconds": command.gpuEndTime, "timed": launch.timed]
        var paths: [String: String] = [:]; var passed = true
        for (tensor, buffer) in zip(manifest.tensor_abi, buffers) {
            let length = tensor.shape.reduce(1, *) * 4
            let name = "launch-\(launch.index)-\(tensor.name).bin"
            let data = try snapshots.capture(buffer.contents(), count: length, name: name, deferred: launch.phase == "cohort")
            paths[tensor.name] = name
            if let before = caseBytes[tensor.name] { passed = passed && data == before }
            else if let (expected, bounds) = caseOracles[tensor.name] {
                data.withUnsafeBytes { raw in
                    for index in expected.indices {
                        let actual = Double(raw.loadUnaligned(fromByteOffset: index * 4, as: Float.self))
                        if !actual.isFinite || abs(actual - expected[index]) > bounds[index] { passed = false }
                    }
                }
            }
        }
        var result: [String: Any] = ["index": launch.index, "role": launch.role, "phase": launch.phase,
            "input_case_id": launch.input_case_id, "command_buffer": commandObservation,
            "buffer_paths": paths, "preflight_guard_passed": passed]
        if let counter = counter {
            guard let data = try counter.resolveCounterRange(0..<2), data.count == 16 else { throw Refusal(description: "GPUTimestamp result is not two UInt64 samples") }
            let ticks = data.withUnsafeBytes { [$0.loadUnaligned(fromByteOffset: 0, as: UInt64.self), $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self)] }
            try require(ticks[0] > 0 && ticks[1] > ticks[0] && ticks[1] != UInt64.max, "invalid GPUTimestamp samples")
            result["profile"] = ["counter_set": "Timestamp", "counter": "GPUTimestamp", "sampling_boundary": "compute_stage",
                "units": "raw_device_timestamp_units", "sample_indices": [0, 1], "resolved_bytes": 16,
                "timestamps": ticks, "command_buffer": commandObservation]
        }
        return result
    }
}
#if !SNAPSHOT_TESTS
var report: [String: Any] = ["schema_version": 1, "status": "failed", "source_library_rebuilt": false]
var snapshots: OwnedSnapshots? = nil
var observations: [[String: Any]] = []
do {
    let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
    let flushAfter = try snapshotFlushPlan(request.launches, participants: request.participants, limit: snapshotPayloadLimit)
    let snapshotWriter = OwnedSnapshots(directory: URL(fileURLWithPath: request.output_directory, isDirectory: true))
    snapshots = snapshotWriter
    guard let device = MTLCreateSystemDefaultDevice() else { throw Refusal(description: "Metal device unavailable") }
    let host = ["device_name": device.name, "device_registry_id": String(device.registryID),
                "operating_system": ProcessInfo.processInfo.operatingSystemVersionString, "target": request.expected_host["target"] ?? ""]
    try require(host == request.expected_host, "admitted Metal host differs")
    let family: MTLGPUFamily = host["target"] == "apple_gpu_family7" ? .apple7 : .apple8
    try require(["apple_gpu_family7", "apple_gpu_family8"].contains(host["target"]!) && device.supportsFamily(family), "exact Metal target unsupported")
    report["host"] = host
    var prepared: [String: Prepared] = [:]
    for participant in request.participants {
        try require(participant.manifest.target == host["target"] && prepared[participant.role] == nil, "participant target or role differs")
        prepared[participant.role] = try Prepared(participant, cases: request.cases, device: device)
    }
    report["module_loads"] = prepared.count
    guard let queue = device.makeCommandQueue() else { throw Refusal(description: "Metal command queue unavailable") }
    var preflightPassed = true
    for (index, launch) in request.launches.enumerated() {
        try require(launch.index == index, "declared launch sequence differs")
        if launch.phase != "preflight" && !preflightPassed { continue }
        guard let participant = prepared[launch.role] else { throw Refusal(description: "unknown launch participant") }
        let observation = try participant.observe(launch, queue: queue, snapshots: snapshotWriter)
        observations.append(observation)
        if flushAfter.contains(index) { try snapshotWriter.flush() }
        if launch.phase == "preflight" { preflightPassed = preflightPassed && (observation["preflight_guard_passed"] as? Bool == true) }
    }
    try snapshotWriter.flush()
    report["archive_miss_policy"] = "failOnBinaryArchiveMiss"
    report["status"] = "completed"
} catch {
    let (original, flushError) = flushAfterFailure(snapshots, original: error)
    report["error"] = errorDetails(original)
    if let flushError = flushError { report["snapshot_flush_error"] = errorDetails(flushError) }
}
report["snapshot_persistence"] = ["condition": "owned_snapshots_written_at_cohort_end",
    "pending_payload_limit_bytes": snapshotPayloadLimit, "peak_pending_payload_bytes": snapshots?.peakPendingBytes ?? 0,
    "failed_writes": snapshots?.failedWrites ?? []] as [String: Any]
report["launches"] = observations
print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), encoding: .utf8)!)
#endif
