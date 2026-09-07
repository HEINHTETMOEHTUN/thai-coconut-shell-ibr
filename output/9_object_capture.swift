import Foundation
import RealityKit

@main
struct CoconutShellObjectCapture {

    static func main() async {

        // ------------------------------------------------
        // PATHS
        // ------------------------------------------------

        let currentDirectory = FileManager.default.currentDirectoryPath

        let inputFolder = URL(
            fileURLWithPath: currentDirectory
        )
        .deletingLastPathComponent()
        .appendingPathComponent("reconstruction_images")

        let outputFile = URL(
            fileURLWithPath: currentDirectory
        )
        .appendingPathComponent("coconut_shell_3d.usdz")


        print("======================================")
        print(" Thai Coconut-Shell 3D Reconstruction")
        print("======================================")

        print("")
        print("Input:")
        print(inputFolder.path)

        print("")
        print("Output:")
        print(outputFile.path)


        // ------------------------------------------------
        // CHECK INPUT FOLDER
        // ------------------------------------------------

        guard FileManager.default.fileExists(
            atPath: inputFolder.path
        ) else {

            print("")
            print("❌ reconstruction_images folder not found.")
            return
        }


        // ------------------------------------------------
        // COUNT INPUT IMAGES
        // ------------------------------------------------

        do {

            let files = try FileManager.default.contentsOfDirectory(
                at: inputFolder,
                includingPropertiesForKeys: nil
            )

            let imageExtensions = [
                "jpg",
                "jpeg",
                "png",
                "heic"
            ]

            let images = files.filter {

                imageExtensions.contains(
                    $0.pathExtension.lowercased()
                )
            }

            print("")
            print("📸 Images found: \(images.count)")

            if images.count < 10 {

                print("")
                print("⚠️ Very few images.")
                print("More overlapping photographs usually improve reconstruction.")
            }

        } catch {

            print("❌ Cannot read image folder.")
            print(error)
            return
        }


        // ------------------------------------------------
        // CHECK OBJECT CAPTURE SUPPORT
        // ------------------------------------------------

        guard PhotogrammetrySession.isSupported else {

            print("")
            print("❌ Object Capture is not supported on this Mac.")
            return
        }


        // ------------------------------------------------
        // CONFIGURATION
        // ------------------------------------------------

        var config = PhotogrammetrySession.Configuration()

        // Images weren't necessarily captured
        // in filename order.
        config.sampleOrdering = .unordered

        // Useful for detailed coconut-shell texture.
        config.featureSensitivity = .high


        // ------------------------------------------------
        // START PHOTOGRAMMETRY
        // ------------------------------------------------

        do {

            print("")
            print("Creating photogrammetry session...")

            let session = try PhotogrammetrySession(
                input: inputFolder,
                configuration: config
            )


            // ------------------------------------------------
            // REQUEST MODEL
            // ------------------------------------------------

            print("Starting 3D reconstruction...")
            print("This may take some time.")
            print("")

            try session.process(
                requests: [

                    .modelFile(
                        url: outputFile,
                        detail: .medium
                    )

                ]
            )


            // ------------------------------------------------
            // MONITOR OUTPUT
            // ------------------------------------------------

            for try await output in session.outputs {

                switch output {


                // ----------------------------
                // INPUT PROCESSING
                // ----------------------------

                case .inputComplete:

                    print("")
                    print("✅ All photographs loaded.")


                // ----------------------------
                // PROGRESS
                // ----------------------------

                case .requestProgress(
                    _,
                    let fraction
                ):

                    let percentage = Int(
                        fraction * 100
                    )

                    print(
                        "Reconstruction progress: \(percentage)%"
                    )


                // ----------------------------
                // MODEL COMPLETED
                // ----------------------------

                case .requestComplete(
                    _,
                    let result
                ):

                    print("")
                    print("✅ Model request completed!")

                    switch result {

                    case .modelFile(let url):

                        print(
                            "3D model created:"
                        )

                        print(
                            url.path
                        )

                    default:
                        break
                    }


                // ----------------------------
                // BAD IMAGE
                // ----------------------------

                case .invalidSample(
                    let id,
                    let reason
                ):

                    print(
                        "⚠️ Invalid photograph \(id)"
                    )

                    print(
                        "Reason: \(reason)"
                    )


                // ----------------------------
                // REQUEST ERROR
                // ----------------------------

                case .requestError(
                    _,
                    let error
                ):

                    print("")
                    print(
                        "❌ Reconstruction request error:"
                    )

                    print(error)


                // ----------------------------
                // EVERYTHING FINISHED
                // ----------------------------

                case .processingComplete:

                    print("")
                    print("======================================")
                    print("🎉 3D RECONSTRUCTION COMPLETED")
                    print("======================================")

                    print("")
                    print(
                        "Saved as:"
                    )

                    print(
                        outputFile.path
                    )

                    return


                default:
                    break
                }
            }


        } catch {

            print("")
            print("❌ Reconstruction failed.")
            print(error)
        }
    }
}