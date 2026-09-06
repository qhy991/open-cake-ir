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

func execute() throws {
    try require(CommandLine.arguments.count == 3 &&
                ["--compile-only", "--run"].contains(CommandLine.arguments[1]),
                "usage: metal-runner --compile-only|--run /absolute/manifest.json")
    let compileOnly = CommandLine.arguments[1] == "--compile-only"
    let manifestPath = CommandLine.arguments[2]
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
    try require(manifest.target == "apple_gpu_family8" && manifest.source_language == "metal" &&
                manifest.compiler == "MTLDevice.makeLibrary", "unsupported exact Metal route")
    try require(manifest.language_standard == "metal2.3" && !manifest.fast_math_enabled &&
                manifest.threadgroup_memory_bytes == 0 &&
                manifest.execution_model == "serial_program_tile" &&
                manifest.active_threads_per_threadgroup == 1,
                "unsupported Metal language, math, memory or execution commitment")
    let grid = try size(manifest.threadgroups_per_grid, "threadgroups_per_grid")
    let threads = try size(manifest.threads_per_threadgroup, "threads_per_threadgroup")
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
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw Refusal(description: "no Metal device available")
    }
    try require(manifest.device_names.contains(device.name) && device.supportsFamily(.apple8) &&
                !device.supportsFamily(.apple9),
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
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: manifest.entry_point) else {
        throw Refusal(description: "emitted entry point unavailable")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    try require(pipeline.maxTotalThreadsPerThreadgroup >= 32 &&
                pipeline.threadExecutionWidth == 32 &&
                pipeline.staticThreadgroupMemoryLength == 0,
                "compiled pipeline violates thread/memory commitments")
    var receipt: [String: Any] = ["status": "compiled", "device": device.name,
        "target": manifest.target, "language_standard": manifest.language_standard,
        "math_mode": "safe", "math_functions": "precise",
        "thread_execution_width": pipeline.threadExecutionWidth,
        "threads_per_threadgroup": manifest.threads_per_threadgroup,
        "threadgroups_per_grid": manifest.threadgroups_per_grid,
        "execution_model": manifest.execution_model,
        "active_threads_per_threadgroup": manifest.active_threads_per_threadgroup]
    if !compileOnly {
        var buffers: [MTLBuffer] = []
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
        guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw Refusal(description: "Metal command creation failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try require(command.status == .completed && command.error == nil,
                    "Metal command failed: \(String(describing: command.error))")
        for (buffer, spec) in zip(buffers, manifest.buffers) {
            let bytes = Data(bytes: buffer.contents(), count: spec.size_bytes)
            try bytes.write(to: URL(fileURLWithPath: spec.output_path), options: .withoutOverwriting)
        }
        receipt["status"] = "completed"
        receipt["command_status"] = "completed"
    }
    let result = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    FileHandle.standardOutput.write(result)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

do {
    try execute()
} catch {
    FileHandle.standardError.write(Data("Metal runtime refusal: \(error)\n".utf8))
    exit(1)
}
