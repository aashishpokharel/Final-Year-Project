import "./PredictionResult.css";
function PredictionResult({ value }) {
  return (
    <>
      <div className="main">
        <div className="result-block">
          {JSON.parse(value).map((item, index) => {
            let contentClass = "content-1";
            let guess = "Best Guess";
            if (index === 0) {
              contentClass = "content-1";
              guess = "Best Guess: ";
            } else if (index === 1) {
              contentClass = "content-2";
              guess = "Second Guess: ";
            } else if (index === 2) {
              contentClass = "content-3";
              guess = "Third Guess: ";
            }

            return (
              <div className={contentClass}>
                <h1>
                  {guess} {item[0]}
                </h1>
                <div className="result">
                  Confidence: {(item[1] * 100).toFixed(2)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

export default PredictionResult;
