import React from "react";

function PredictionBoard(props) {
  const { children, clickHandler } = props;

  return (
    <>
      <button className="predict-button" onClick={clickHandler}>
        {children}
      </button>
    </>
  );
}

export default PredictionBoard;
