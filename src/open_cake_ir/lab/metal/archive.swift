// Build or reload a compiled archive. No command buffers or dispatch APIs.
import Foundation
import Metal

struct Request: Decodable {
    let action: String
    let target: String
    let expected_device_names: [String]
    let entry_point: String?
    let archive_path: String?
    let source_path: String?
    let expected_host: [String: String]?
}
struct Refusal: Error, CustomStringConvertible { let description: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Refusal(description: message) }
}
var stage = "request"
var report: [String: Any] = ["schema_version": 1, "status": "failed", "dispatches": 0]
do {
    let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let request = try JSONDecoder().decode(Request.self, from: data)
    try require(["build", "reload", "inspect"].contains(request.action), "unsupported archive action")
    stage = "host_admission"
    guard let device = MTLCreateSystemDefaultDevice() else { throw Refusal(description: "Metal device unavailable") }
    let family: MTLGPUFamily
    switch request.target {
    case "apple_gpu_family7": family = .apple7
    case "apple_gpu_family8": family = .apple8
    default: throw Refusal(description: "unsupported exact Metal target")
    }
    try require(request.expected_device_names.contains(device.name) && device.supportsFamily(family), "exact Metal device differs")
    let host = ["device_name": device.name, "device_registry_id": String(device.registryID),
                "operating_system": ProcessInfo.processInfo.operatingSystemVersionString, "target": request.target]
    if let expected = request.expected_host { try require(host == expected, "compiled archive host admission differs") }
    report["host"] = host
    if request.action == "inspect" {
        try require(request.source_path == nil && request.archive_path == nil && request.entry_point == nil,
                    "host inspection must not receive source or archive input")
        let stageSampling = device.supportsCounterSampling(.atStageBoundary)
        let timestampSet = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        let timestampAvailable = timestampSet?.counters.contains(where: { $0.name == "GPUTimestamp" }) ?? false
        report["profiling"] = ["compute_stage_sampling": stageSampling, "gpu_timestamp_counter": timestampAvailable]
        try require(stageSampling && timestampAvailable, "compute-stage GPUTimestamp sampling unavailable")
        report["status"] = "completed"
        report["stage"] = stage
        let payload = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(data: payload, encoding: .utf8)!)
        exit(0)
    }
    guard let archivePath = request.archive_path, let entryPoint = request.entry_point else {
        throw Refusal(description: "archive build/reload requires artifact path and entry point")
    }
    let archiveURL = URL(fileURLWithPath: archivePath)
    let archive: MTLBinaryArchive
    let library: MTLLibrary
    if request.action == "build" {
        stage = "source_compile"
        guard let sourcePath = request.source_path else { throw Refusal(description: "build requires lowered source") }
        let source = try String(contentsOfFile: sourcePath, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version2_3
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        library = try device.makeLibrary(source: source, options: options)
        archive = try device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
    } else {
        stage = "archive_load"
        try require(request.source_path == nil, "archive reload must not receive source")
        let archiveDescriptor = MTLBinaryArchiveDescriptor()
        archiveDescriptor.url = archiveURL
        archive = try device.makeBinaryArchive(descriptor: archiveDescriptor)
        library = try device.makeLibrary(URL: archiveURL)
    }
    stage = "function_lookup"
    guard let function = library.makeFunction(name: entryPoint) else { throw Refusal(description: "archived entry point unavailable") }
    let descriptor = MTLComputePipelineDescriptor()
    descriptor.computeFunction = function
    if request.action == "build" {
        stage = "archive_serialize"
        try require(!FileManager.default.fileExists(atPath: archiveURL.path), "archive already exists")
        try archive.addComputePipelineFunctions(descriptor: descriptor)
        try archive.serialize(to: archiveURL)
    }
    stage = "strict_pipeline_load"
    descriptor.binaryArchives = [archive]
    let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
        options: [.failOnBinaryArchiveMiss], reflection: nil)
    // The SIMD width stays 32. Consecutive groups raise the threadgroup size and bring
    // their own static threadgroup storage; Python checks both against the manifest.
    try require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 32
                && pipeline.staticThreadgroupMemoryLength >= 0
                && pipeline.staticThreadgroupMemoryLength <= 32768,
                "pipeline resources differ from admitted SIMD launch")
    report["pipeline"] = ["thread_execution_width": pipeline.threadExecutionWidth,
        "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
        "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength]
    report["archive_miss_policy"] = "failOnBinaryArchiveMiss"
    report["source_library_rebuilt"] = request.action == "build"
    report["status"] = "completed"
} catch {
    let native = error as NSError
    report["error"] = ["domain": native.domain, "code": native.code,
        "description": (error as? Refusal)?.description ?? native.localizedDescription]
}
report["stage"] = stage
let payload = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
print(String(data: payload, encoding: .utf8)!)
