import DrawingBoard from "./Components/DrawingBoard/DrawingBoard";
import ImageUpload from "./Components/ImageUpload/ImageUpload";
import PredictionBoard from "./Components/PredictionBoard/PredictionBoard";
function App() {
  return (
    <>
      <PredictionBoard />
      <DrawingBoard />
      <ImageUpload />
    </>
  );
}

export default App;
