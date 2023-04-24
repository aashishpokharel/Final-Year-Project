import React from "react";
import "./PredictionBoard.css";

function PredictionBoard() {
  return (
    <>
      <div className="main">
        <div className="result-block">
          <div className="content-1">
            <h1>1</h1>
            <div className="result">Confidence:93%</div>
          </div>
          <div className="content-2">
            <h1>1</h1>
            <div className="result">Confidence:23%</div>
          </div>
          <div className="content-3">
            <h1>1</h1>
            <div className="result">Confidence:10%</div>
          </div>
        </div>
      </div>
    </>
  );
}

export default PredictionBoard;
