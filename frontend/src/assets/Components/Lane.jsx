import { useState } from "react";
import ObstacleSVG from "../obstacle.svg";
import CarSVG from "../car.svg";
import Empty from "./Empty";

function Lane() {
  const [cars, setCars] = useState([
    [<Empty key="0" />, 0],
    [<Empty key="1" />, 0],
    [<Empty key="2" />, 0],
    [<Empty key="3" />, 0],
    [<Empty key="4" />, 0],
    [<Empty key="5" />, 0],
  ]);

  function getIndex(height) {
    return 5 - Math.floor(height / (758 / 6));
  }

  function placeCar(event) {
    const carIndex = getIndex(event.clientY);

    setCars((oldValue) => {
      let newValue = [...oldValue];
      if (newValue[carIndex][1] === 0 || newValue[carIndex][1] === 2) {
        newValue[carIndex] = [
          <img key={carIndex} className="car" src={CarSVG} />,
          1,
        ];
      } else {
        newValue[carIndex] = [<Empty key={carIndex} />, 0];
      }
      return newValue;
    });
  }

  function placeObstacle(event) {
    event.preventDefault();
    const obstacleIndex = getIndex(event.clientY);

    setCars((oldValue) => {
      let newValue = [...oldValue];
      if (
        newValue[obstacleIndex][1] === 0 ||
        newValue[obstacleIndex][1] === 1
      ) {
        newValue[obstacleIndex] = [
          <img key={obstacleIndex} className="car" src={ObstacleSVG} />,
          2,
        ];
      } else {
        newValue[obstacleIndex] = [<Empty key={obstacleIndex} />, 0];
      }
      return newValue;
    });
  }

  return (
    <div onContextMenu={placeObstacle} onClick={placeCar} className="lane">
      {cars.map((value) => value[0])}
    </div>
  );
}

export default Lane;
