import "./PredictionResult.css";
function PredictionResult({ value, prob }) {
  const myValue = parseFloat(prob.slice(1, -1)).toFixed(5);

  return (
    <>
      <div className="main">
        <div className="result-block">
          <div className="content-1">
            <h1>Best Guess: {value[0]}</h1>
            <div className="result">Confidence: {myValue * 100}%</div>
          </div>
          {/* <div className="content-2">
            <h1>{value[1]}</h1>
            <div className="result">
              Confidence: {value[1].confidence}%
            </div>
          </div>
          <div className="content-3">
            <h1>{value[2]}</h1>
            <div className="result">
              Confidence: {value[2].confidence}%
            </div>
          </div> */}
        </div>
      </div>
    </>
  );
}

export default PredictionResult;
