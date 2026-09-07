import Foundation
import RealityKit

// =========================================================
// RESEARCH EVALUATION — 3D RECONSTRUCTION
// =========================================================
// Compares:
//   A. Horizontal ring only
//   B. Horizontal + upper views
//
// Objective measurements written automatically:
//   - input image count
//   - reconstruction time
//   - USDZ file size
//   - success/failure
//
// Visual completeness is evaluated separately with the
// provided CSV template (front/right/back/left/top views).
// =========================================================

struct ReconstructionResult {
    let method: String
    let imageCount: Int
    let runtimeSeconds: Double
    let fileSizeMB: Double
    let outputPath: String
    let success: Bool
}

@main
struct ReconstructionComparison {

    static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic"]

    static func countImages(in folder: URL) -> Int {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil
        ) else {
            return 0
        }

        return files.filter {
            imageExtensions.contains($0.pathExtension.lowercased())
        }.count
    }

    static func fileSizeMB(_ url: URL) -> Double {
        guard
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
            let bytes = attributes[.size] as? NSNumber
        else {
            return 0.0
        }

        return bytes.doubleValue / (1024.0 * 1024.0)
    }

    static func runReconstruction(
        method: String,
        inputFolder: URL,
        outputFile: URL
    ) async -> ReconstructionResult {

        let imageCount = countImages(in: inputFolder)

        print("\n====================================================")
        print("\(method)")
        print("====================================================")
        print("Input:  \(inputFolder.path)")
        print("Images: \(imageCount)")
        print("Output: \(outputFile.path)")

        guard imageCount >= 3 else {
            print("❌ Not enough images.")
            return ReconstructionResult(
                method: method,
                imageCount: imageCount,
                runtimeSeconds: 0,
                fileSizeMB: 0,
                outputPath: outputFile.path,
                success: false
            )
        }

        if FileManager.default.fileExists(atPath: outputFile.path) {
            try? FileManager.default.removeItem(at: outputFile)
        }

        var config = PhotogrammetrySession.Configuration()
        config.sampleOrdering = .unordered
        config.featureSensitivity = .high

        let start = Date()

        do {
            let session = try PhotogrammetrySession(
                input: inputFolder,
                configuration: config
            )

            try session.process(
                requests: [
                    .modelFile(
                        url: outputFile,
                        detail: .medium
                    )
                ]
            )

            for try await output in session.outputs {
                switch output {

                case .inputComplete:
                    print("✅ Input loading complete")

                case .requestProgress(_, let fraction):
                    let percent = Int(fraction * 100)
                    print("Progress: \(percent)%")

                case .invalidSample(let id, let reason):
                    print("⚠️ Invalid image \(id): \(reason)")

                case .requestError(_, let error):
                    print("❌ Reconstruction request error: \(error)")

                case .processingComplete:
                    let runtime = Date().timeIntervalSince(start)
                    let size = fileSizeMB(outputFile)
                    let exists = FileManager.default.fileExists(atPath: outputFile.path)

                    print("✅ Finished in \(String(format: \"%.2f\", runtime)) seconds")
                    print("USDZ size: \(String(format: \"%.2f\", size)) MB")

                    return ReconstructionResult(
                        method: method,
                        imageCount: imageCount,
                        runtimeSeconds: runtime,
                        fileSizeMB: size,
                        outputPath: outputFile.path,
                        success: exists
                    )

                default:
                    break
                }
            }

        } catch {
            print("❌ Reconstruction failed: \(error)")
        }

        let runtime = Date().timeIntervalSince(start)
        return ReconstructionResult(
            method: method,
            imageCount: imageCount,
            runtimeSeconds: runtime,
            fileSizeMB: fileSizeMB(outputFile),
            outputPath: outputFile.path,
            success: false
        )
    }

    static func csvEscape(_ text: String) -> String {
        let escaped = text.replacingOccurrences(of: "\"", with: "\"\"")
        return "\"\(escaped)\""
    }

    static func main() async {
        guard PhotogrammetrySession.isSupported else {
            print("❌ RealityKit Object Capture is not supported on this Mac.")
            return
        }

        // IMPORTANT: run this executable while Terminal is inside project/output.
        let outputDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let projectRoot = outputDirectory.deletingLastPathComponent()

        let horizontalFolder = projectRoot
            .appendingPathComponent("images")
            .appendingPathComponent("horizontal")

        let combinedFolder = projectRoot
            .appendingPathComponent("reconstruction_images")

        let evaluationFolder = projectRoot
            .appendingPathComponent("evaluation")
            .appendingPathComponent("reconstruction")

        do {
            try FileManager.default.createDirectory(
                at: evaluationFolder,
                withIntermediateDirectories: true
            )
        } catch {
            print("❌ Could not create evaluation folder: \(error)")
            return
        }

        let horizontalOutput = evaluationFolder
            .appendingPathComponent("horizontal_only.usdz")

        let combinedOutput = evaluationFolder
            .appendingPathComponent("horizontal_plus_upper.usdz")

        print("====================================================")
        print(" 3D RECONSTRUCTION RESEARCH EVALUATION")
        print("====================================================")
        print("Detail level: MEDIUM for both models (fair comparison)")

        let horizontalResult = await runReconstruction(
            method: "Horizontal Only",
            inputFolder: horizontalFolder,
            outputFile: horizontalOutput
        )

        let combinedResult = await runReconstruction(
            method: "Horizontal + Upper",
            inputFolder: combinedFolder,
            outputFile: combinedOutput
        )

        let results = [horizontalResult, combinedResult]
        let csvURL = evaluationFolder.appendingPathComponent("reconstruction_results.csv")

        var csv = "Method,Input_Images,Runtime_s,File_Size_MB,Success,Output_File\n"
        for result in results {
            csv += "\(csvEscape(result.method)),"
            csv += "\(result.imageCount),"
            csv += "\(String(format: \"%.6f\", result.runtimeSeconds)),"
            csv += "\(String(format: \"%.6f\", result.fileSizeMB)),"
            csv += "\(result.success ? 1 : 0),"
            csv += "\(csvEscape(result.outputPath))\n"
        }

        do {
            try csv.write(to: csvURL, atomically: true, encoding: .utf8)
        } catch {
            print("❌ Could not write CSV: \(error)")
        }

        print("\n====================================================")
        print(" RECONSTRUCTION COMPARISON COMPLETE")
        print("====================================================")
        for result in results {
            print(
                "\(result.method): " +
                "images=\(result.imageCount), " +
                "time=\(String(format: \"%.2f\", result.runtimeSeconds))s, " +
                "size=\(String(format: \"%.2f\", result.fileSizeMB))MB, " +
                "success=\(result.success)"
            )
        }
        print("\nCSV saved to:")
        print(csvURL.path)
        print("\nNext: open both USDZ files and score front/right/back/left/top completeness using the visual template.")
    }
}
