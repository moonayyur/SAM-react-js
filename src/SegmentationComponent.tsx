import { useCallback, useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import * as tf from "@tensorflow/tfjs";

const decoderModelPath = "/models/sam_vit_b_01ec64.decoder.onnx";
const encoderModelPath =
  "/models/sam_vit_b_01ec64.encoder.preprocess.quant.onnx";

export default function SegmentationComponent() {
  const [status, setStatus] = useState<string>("-");
  const [decoderResults, setDecoderResults] =
    useState<ort.InferenceSession.OnnxValueMapType>();

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [imageEmbeddings, setImageEmbeddings] = useState<ort.Tensor | null>(
    null,
  );
  const [imageDataImage, setImageDataImage] = useState<ImageData>();

  const canvas = canvasRef.current;
  const context = canvas?.getContext("2d");

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      const imgURL = URL.createObjectURL(file);
      loadImage(imgURL);
    }
  }

  function loadImage(imgURL: string) {
    const img = new Image();
    img.onload = () => imageEncoder(imgURL);
    img.src = imgURL;
  }

  const displayCanvas = useCallback(
    (x?: number, y?: number) => {
      if (!canvas || !imageDataImage || !context) {
        return;
      }
      context.clearRect(0, 0, canvas.width, canvas.height);

      canvas.width = imageDataImage.width;
      canvas.height = imageDataImage.height;
      context.putImageData(imageDataImage, 0, 0);
      if (x && y) {
        context.fillStyle = "green";
        context.fillRect(x, y, 10, 10);
      }
    },
    [canvas, context, imageDataImage],
  );

  useEffect(() => {
    displayCanvas();
  }, [displayCanvas, imageDataImage]);

  useEffect(() => {
    console.log("Image Data Image", imageDataImage);
  }, [imageDataImage]);

  const maskDecoder = useCallback(
    async (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvas || !context || !imageEmbeddings) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      console.log("Point input :", x, y);
      setStatus(
        `Point (${x}, ${y}). Downloading the decoder model if needed and generating mask...`,
      );

      console.log("Canvas dimensions:", canvas!.width, canvas!.height);

      displayCanvas(x, y);

      //ort inputs preparation
      const pointCoords = new ort.Tensor(
        new Float32Array([x, y, 0, 0]), //padding concatenated to coordinates
        [1, 2, 2],
      );
      const pointLabels = new ort.Tensor(new Float32Array([0, -1]), [1, 2]); //padding point concatenated to positive input point
      const maskInput = new ort.Tensor( //must be supplied even if no mask is available
        new Float32Array(256 * 256),
        [1, 1, 256, 256],
      );
      const hasMask = new ort.Tensor(new Float32Array([0]), [1]); //0 for no mask input
      const originalImageSize = new ort.Tensor(
        new Float32Array([canvas!.height, canvas!.width]),
        [2],
      );
      const decodingFeeds = {
        image_embeddings: imageEmbeddings!,
        point_coords: pointCoords,
        point_labels: pointLabels,
        mask_input: maskInput,
        has_mask_input: hasMask,
        orig_im_size: originalImageSize,
      };

      const decodingSession =
        await ort.InferenceSession.create(decoderModelPath);

      console.log("Decoder session", decodingSession);
      console.log("Generating mask...");

      const start = Date.now();

      const results = await decodingSession.run(decodingFeeds);

      console.log("Generated mask:", results);

      const maskImageData = results.masks.toImageData();
      context!.globalAlpha = 0.5;
      const imageBitmap = await createImageBitmap(maskImageData);
      console.log("imageBitmap", imageBitmap.width, imageBitmap.height);
      context!.drawImage(imageBitmap, 0, 0);
      setDecoderResults(results);

      const end = Date.now();

      console.log(`Generating masks took : ${(end - start) / 1000} seconds`);
      setStatus(`Mask generated. Click on the image to generate a new mask`);
    },
    [canvas, context, displayCanvas, imageEmbeddings],
  );

  const imageEncoder = useCallback(async (imgURL: string) => {
    const img = new Image();
    img.src = imgURL;

    setStatus(
      `Image size ${img.width}x${img.height}. Downloading the encoder model if not cached and generating embedding...`,
    );

    const resizedTensor = await ort.Tensor.fromImage(img, {
      resizedWidth: 1024,
      resizedHeight: 1024,
    });
    const resizedImage = resizedTensor.toImageData();
    let imageDataTensor = await ort.Tensor.fromImage(resizedImage);
    setImageDataImage(imageDataTensor.toImageData());
    console.log("Image Data Tensor", imageDataTensor);

    let tf_tensor = tf.tensor(
      imageDataTensor.data,
      imageDataTensor.dims as number[],
    );
    tf_tensor = tf_tensor.reshape([3, 1024, 1024]);
    tf_tensor = tf_tensor.transpose([1, 2, 0]).mul(255);

    imageDataTensor = new ort.Tensor(
      tf_tensor.dataSync(),
      tf_tensor.shape,
    ) as ort.TypedTensor<"float32">;

    const session = await ort.InferenceSession.create(encoderModelPath);

    console.log("Encoder Session", session);

    const feeds = { input_image: imageDataTensor };

    console.log("Generating image embedding...");

    const start = Date.now();

    const results = await session.run(feeds);
    console.log("Encoding result:", results);
    setImageEmbeddings(results.image_embeddings);

    const end = Date.now();

    const timeTaken = (end - start) / 1000;
    console.log(`Generating image embedding took : ${timeTaken} seconds`);
    setStatus(
      `Embedding generated in : ${timeTaken} seconds. Click on the image to generate a mask`,
    );
  }, []);

  function displayResults() {
    if (decoderResults) {
      console.log(decoderResults.iou_predictions);
      return (
        <div>
          <div className="text-xl pl-2 pt-2">Masks :</div>
          <div>Data : {decoderResults.masks.data.slice(0, 20).toString()}</div>
          <div>Size : {decoderResults.masks.size}</div>
          <div>Dims : {decoderResults.masks.dims.toString()}</div>

          <div className="text-xl pl-2 pt-2">Low resolution masks :</div>
          <div>
            Data : {decoderResults.low_res_masks.data.slice(0, 20).toString()}
          </div>
          <div>Size : {decoderResults.low_res_masks.size}</div>
          <div>Dims : {decoderResults.low_res_masks.dims.toString()}</div>

          <div className="text-xl pl-2 pt-2">IoU predictions :</div>
          <div>Data : {decoderResults.iou_predictions.data.toString()}</div>
          <div>Size : {decoderResults.iou_predictions.size}</div>
          <div>Dims : {decoderResults.iou_predictions.dims.toString()}</div>
        </div>
      );
    }
    return <div>-</div>;
  }

  return (
    <div className="w-screen px-10 flex flex-col">
      <div className="p-5 text-5xl">Segment Anything with tfjs and ort</div>
      <div className="p-3">
        <input type="file" onChange={handleFileChange} />
      </div>
      <div className="p-3">
        <div className="text-2xl">Status :</div>
        <div>{status}</div>
      </div>
      <div className="p-3">
        <canvas ref={canvasRef} onClick={maskDecoder}></canvas>
      </div>
      <div className="p-3">
        <div className="text-2xl">Generated masks :</div>
        <div className="">{displayResults()}</div>
      </div>
    </div>
  );
}
