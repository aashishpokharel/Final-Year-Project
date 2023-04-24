import React, { useState } from "react";
import "./ImageUpload.css";
import axios from "axios";
import { useRef } from "react";
import PredictionBoard from "../PredictionBoard/PredictionBoard";
import PredictionResult from "../PredictionResult/PredictionResult";
function ImageUpload() {
  const [imageURL, setImageURL] = useState(null);
  const [predictedValue, setPredictedValue] = useState(null);
  const [image, setImage] = useState(null);
  const imageRef = useRef();

  const handleUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", image);
    try {
      const response = await axios.post(
        "http://localhost:8000/image-upload",
        formData
      );
      setPredictedValue(response.data.Prediction[1]); ///because the return type is an array with 1st index as '['
    } catch (error) {
      console.log(error);
    }
  };

  const getFileInfo = (e) => {
    const { files } = e.target;

    //to display the chosen file in the screen
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0]);
      setImageURL(url);
      setImage(files[0]);
    } else {
      setImageURL(null);
    }
  };

  return (
    <div>
      <div className="mainWrapper">
        <div className="mainContent">
          <div className="imageHolder">
            {imageURL && (
              <div className="uploadedImage">
                <img
                  src={imageURL}
                  alt="uploaded-img"
                  crossOrigin="anonymous"
                  ref={imageRef}
                />
              </div>
            )}
          </div>
          {predictedValue && <PredictionResult value={predictedValue} />}
        </div>
      </div>
      <div className="inputHolder">
        {!imageURL && (
          /*isCanvasEmpty &&*/ <div className="uploadInput">
            <input
              type="file"
              name="file"
              accept="image/*"
              capture="camera"
              onChange={getFileInfo}
            />
          </div>
        )}
        {imageURL && (
          <PredictionBoard className="uploadButton" clickHandler={handleUpload}>
            Predict
          </PredictionBoard>
        )}
      </div>
    </div>
  );
}

export default ImageUpload;
