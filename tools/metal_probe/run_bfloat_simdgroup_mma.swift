import Darwin
import Foundation
import Metal

private let probeKind = "open_cake_metal_bfloat_simdgroup_mma_dispatch_v1"
private let entryPoint = "open_cake_bfloat_simdgroup_mma_probe"
private let matrixExtent = 8
private let matrixElementCount = 64

private enum ProbeError: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

private func require<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else {
        throw ProbeError.failed(message)
    }
    return value
}

private func runProbe(metallibPath: String) throws -> [String: Any] {
    let metallibURL = URL(fileURLWithPath: metallibPath)
    var isDirectory = ObjCBool(false)
    guard metallibURL.path.hasPrefix("/") else {
        throw ProbeError.failed("metallib path must be absolute")
    }
    guard FileManager.default.fileExists(
        atPath: metallibURL.path,
        isDirectory: &isDirectory
    ), !isDirectory.boolValue else {
        throw ProbeError.failed("metallib path is not a regular file")
    }
    let metallibBytes = try Data(contentsOf: metallibURL, options: .mappedIfSafe)
    guard metallibBytes.count >= 4,
          Array(metallibBytes.prefix(4)) == [0x4d, 0x54, 0x4c, 0x42] else {
        throw ProbeError.failed("metallib is missing the MTLB header")
    }

    let device = try require(MTLCreateSystemDefaultDevice(), "Metal device is unavailable")
    guard device.supportsFamily(.apple9) else {
        throw ProbeError.failed("Metal device does not support Apple GPU family 9")
    }
    let library = try device.makeLibrary(URL: metallibURL)
    let function = try require(
        library.makeFunction(name: entryPoint),
        "metallib entry point is missing"
    )
    let pipeline = try device.makeComputePipelineState(function: function)
    guard pipeline.threadExecutionWidth == 32 else {
        throw ProbeError.failed(
            "probe requires a 32-thread SIMD-group, observed \(pipeline.threadExecutionWidth)"
        )
    }
    guard pipeline.maxTotalThreadsPerThreadgroup >= pipeline.threadExecutionWidth else {
        throw ProbeError.failed("pipeline cannot admit one complete SIMD-group")
    }

    let lhsValues = (0..<matrixElementCount).map { index in
        Float(index / matrixExtent == index % matrixExtent ? 1 : 0)
    }
    let rhsValues = (0..<matrixElementCount).map { Float($0 + 1) }
    let lhsBits = lhsValues.map { UInt16($0.bitPattern >> 16) }
    let rhsBits = rhsValues.map { UInt16($0.bitPattern >> 16) }
    var expected = [Float](repeating: 0, count: matrixElementCount)
    for row in 0..<matrixExtent {
        for column in 0..<matrixExtent {
            for inner in 0..<matrixExtent {
                expected[row * matrixExtent + column] +=
                    lhsValues[row * matrixExtent + inner]
                    * rhsValues[inner * matrixExtent + column]
            }
        }
    }
    let inputByteCount = lhsBits.count * MemoryLayout<UInt16>.stride
    let outputByteCount = matrixElementCount * MemoryLayout<Float>.stride
    let lhs = try lhsBits.withUnsafeBytes { bytes in
        try require(
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: inputByteCount,
                options: .storageModeShared
            ),
            "failed to allocate the lhs Metal buffer"
        )
    }
    let rhs = try rhsBits.withUnsafeBytes { bytes in
        try require(
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: inputByteCount,
                options: .storageModeShared
            ),
            "failed to allocate the rhs Metal buffer"
        )
    }
    let output = try require(
        device.makeBuffer(length: outputByteCount, options: .storageModeShared),
        "failed to allocate the output Metal buffer"
    )
    memset(output.contents(), 0, outputByteCount)

    let queue = try require(device.makeCommandQueue(), "failed to create a Metal command queue")
    let commandBuffer = try require(queue.makeCommandBuffer(), "failed to create a command buffer")
    let encoder = try require(
        commandBuffer.makeComputeCommandEncoder(),
        "failed to create a compute command encoder"
    )
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(lhs, offset: 0, index: 0)
    encoder.setBuffer(rhs, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    let threads = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
    encoder.dispatchThreads(threads, threadsPerThreadgroup: threads)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw ProbeError.failed("Metal command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.failed(
            "Metal command buffer did not complete (status \(commandBuffer.status.rawValue))"
        )
    }

    let pointer = output.contents().bindMemory(
        to: Float.self,
        capacity: matrixElementCount
    )
    let values = Array(
        UnsafeBufferPointer(start: pointer, count: matrixElementCount)
    )
    guard zip(values, expected).allSatisfy({ pair in
        pair.0.isFinite && pair.0 == pair.1
    }) else {
        let firstMismatch = values.indices.first { index in
            !values[index].isFinite || values[index] != expected[index]
        }
        let detail = firstMismatch.map { "index \($0), value \(values[$0])" } ?? "unknown"
        throw ProbeError.failed("BF16 SIMD-group MMA output mismatch: \(detail)")
    }

    return [
        "schema_version": 1,
        "kind": probeKind,
        "status": "passed",
        "kernel_calls": 1,
        "device": [
            "name": device.name,
            "registry_id": device.registryID,
            "supports_apple9": true,
            "has_unified_memory": device.hasUnifiedMemory,
            "max_threadgroup_memory_length": device.maxThreadgroupMemoryLength,
        ],
        "pipeline": [
            "entry_point": entryPoint,
            "thread_execution_width": pipeline.threadExecutionWidth,
            "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
            "static_threadgroup_memory_length": pipeline.staticThreadgroupMemoryLength,
        ],
        "dispatch": [
            "mma_shape": ["m": 8, "n": 8, "k": 8],
            "input_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "input_pattern": "lhs_identity_rhs_row_major_1_to_64",
            "expected_outputs": expected,
            "observed_outputs": values,
            "output_element_count": values.count,
            "all_outputs_match": true,
            "threads_dispatched": pipeline.threadExecutionWidth,
            "threadgroups_dispatched": 1,
            "command_buffer_status": "completed",
        ],
    ]
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw ProbeError.failed("usage: run_bfloat_simdgroup_mma.swift METALLIB")
    }
    let observation = try runProbe(metallibPath: CommandLine.arguments[1])
    guard JSONSerialization.isValidJSONObject(observation) else {
        throw ProbeError.failed("probe observation is not valid JSON")
    }
    let encoded = try JSONSerialization.data(
        withJSONObject: observation,
        options: [.sortedKeys]
    )
    FileHandle.standardOutput.write(encoded)
    FileHandle.standardOutput.write(Data([0x0a]))
} catch {
    let message = "metal probe failed: \(error)\n"
    FileHandle.standardError.write(message.data(using: .utf8)!)
    exit(1)
}
