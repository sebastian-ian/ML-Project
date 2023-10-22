let numberOfLayers = 4;
let largestRow = 9;
let dimensionsArray = [4,5,3,1]; // number of nodes in each layer, 196 input nodes in a 14 x 14 image
let valuesArray = []; //2 dimensional layer, row
let weightsArray = []; // 3 dimensional layer, row, weight
let biasArray = []; // same as weights
let slopeArray = [] // same as weights
let biasSlopeArray = [];
let expectedValue = 7;
let expectedArray = [];//[0,0,0,0,1,0,0,0,0];
let e = 2.7182818284;
let cost;
let learningRate = 0.1;
let pixelsArray = [];
let xPixels = 16;
let yPixels = 16;
let echo = false;
let epoch = 0;


function setup() {
  setArrays();
  createCanvas(800, 800);
  setInputs();
  setWeights();
  //frameRate(1);
  setPixels();
}


function draw() 
{
      for(i = 0; i < 1000; i++)
  {
    setNewInputs();

    background(220);
    //drawnInput();
    cost = propogate(expectedArray);
    //print(cost);
    drawNet();
    backpropogate2();
    //print(weightsArray[3,0,0]);
    //noLoop();
    //print(valuesArray[2][1]);
  }
  epoch++;
  if (epoch > 50)
  keyPressed();

}

function setNewInputs()
{

    expectedArray[0] = 0;
    for(row = 0; row < 4; row++)
    {
      valuesArray[0][row] = float(random(0.0,1.0));
      expectedArray[0] += valuesArray[0][row];
    }
    expectedArray[0] /= 4;
}

function keyPressed()
{
  //cost = propogate(expectedArray);
  //weightsArray[1][1][1] += 1; 
  //let newCost = cost - propogate(expectedArray);
  //weightsArray[1][1][1] -= 1;
  ////print(valuesArray[4,7])
  //print(cost);
  //print(newCost);
  //draw();
  echo = true;
  expectedArray[0] = 0;
  for(row = 0; row < 4; row++)
  {
    valuesArray[0][row] = float(random(0.0,1.0)); //0.05;
    expectedArray[0] += valuesArray[0][row];
  }
  expectedArray[0] /= 4;
  cost = propogate(expectedArray);
  print('expected value: ' + expectedArray[0]);
  print(valuesArray[numberOfLayers - 1][0]);
  background(220);
  drawNet();  
  noLoop();
}

function propogate(expected)
{
  for(let layer = 0; layer < numberOfLayers; layer++)
  {
    for(let weight = 0; weight < dimensionsArray[layer+1]; weight++) // loops through all the rows in a layer, multiplying by weight associated with the first row in the next layer, then increments the weight. This allows the function to go one row at a time in the layer it is propogating to, applying the sigmoid after adding together all the weights.
    {
      valuesArray[layer+1][weight] = 0; // since each weight is associated with a row in the next layer, we are able to use the weight in place of a row. This means that all the rows in the current layer multiply to the weight associated to the same row in the next layer and add it to the correct row in the next layer. 
      for(let row = 0; row < dimensionsArray[layer]; row++)
      {
        valuesArray[layer+1][weight] += valuesArray[layer][row] * weightsArray[layer][row][weight];
        // next layer node += current layer * weights
        // remember the weight has the same dimensions as the next layer 
      }
      valuesArray[layer+1][weight] += biasArray[layer+1][weight];
      /////////////////
      valuesArray[layer+1][weight] = 1 / (1 + (pow(e,-valuesArray[layer+1][weight]))); //sigmoid
      //valuesArray[layer+1][weight] = max(0,min(1,valuesArray[layer+1][weight] * 0.25 + 0.5)); // ReLu
      /////////////////
              //if(echo === true) print('relu' + ': ' + valuesArray[layer+1][weight]);
      // once the weight array progresses, all the values and weights have been added so the sigmoid can be applied before moving on to the next weight.
    }
  }
  if(expected)
  {
    return calculateCost(expected);
  }
}

function calculateCost(expected)
{
  let currentCost = 0;
  for(row = 0; row < dimensionsArray[numberOfLayers-1]; row++) // compare the final layer to the expected values in the final row
  {
    currentCost += pow((valuesArray[numberOfLayers-1][row] - expected[row]),2);
    // add up the differences between the final layer and the expected final layer squared to find the cost, inaccuracy, of the network
  }
  return currentCost;
}


function backpropogate2()
{
  /*
  approximate the derivative of the weight by seeing how a slight change in the weight affects the cost function
  */
  // change the weight and then propogate the array
  // measure the change in the cost function 
  // adjust the weight accordingly. A larger change in the cost function should correspond to a larger change in the weight.
  // loop through all the weights and do this with them all. Don't update the weights until the end.
  for(let layer = numberOfLayers-2; layer >= 0; layer--) // starts on the second to last layer
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      for(let weight = 0; weight < dimensionsArray[layer+1]; weight++) // the number of weights in each node is equal to the number of nodes in the next layer
      {
        let deltaWeight = 0.01
        //if(weightsArray[layer,row,weight] >= 0) print(weightsArray[layer,row,weight]);
        weightsArray[layer][row][weight] += deltaWeight;
        //if(weightsArray[layer,row,weight] >= 0) print(weightsArray[layer,row,weight]);
        let newCost = propogate(expectedArray);
        weightsArray[layer][row][weight] -= deltaWeight;
        let slope = (newCost - cost)/deltaWeight;
        //print(slope);
        slopeArray[layer][row][weight] = slope;
        
        // do the same thing for the bias
        biasArray[layer+1][weight] += deltaWeight;
        //if(weightsArray[layer,row,weight] >= 0) print(weightsArray[layer,row,weight]);
        newCost = propogate(expectedArray);
        biasArray[layer+1][weight] -= deltaWeight;
        slope = (newCost - cost)/deltaWeight;
        //print(slope);
        biasSlopeArray[layer+1][weight] = slope;
      }
    }
  }
  for(let layer = numberOfLayers-2; layer >= 0; layer--) // starts on the second to last layer
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      for(let weight = 0; weight < dimensionsArray[layer+1]; weight++) // the number of weights in each node is equal to the number of nodes in the next layer
      {
        weightsArray[layer][row][weight] = weightsArray[layer][row][weight] + ((slopeArray[layer][row][weight] * (-1)) * learningRate);
        biasArray[layer+1][weight] = biasArray[layer+1][weight] + ((biasSlopeArray[layer+1][weight] * (-1)) * learningRate);
        //print(slopeArray[layer,row,weight]);
      }
    }
  }
}


function backpropogate3()
{
  /*
  find the derivative of a weight to see how it affects the cost function then increase it or decrease it proportionally.
  */
}

function drawNet()
{
  for(let layer = 0; layer < numberOfLayers; layer++)
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      circle((width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,50);
      text(valuesArray[layer][row], (width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,50)
      for(let weight = 0; weight < dimensionsArray[layer-1]; weight++)
      {
        push();
        stroke(weightsArray[layer][row][weight],weightsArray[layer][row][weight]*255*2,weightsArray[layer][row][weight]*255*3);
        line((width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,(width-50)*(layer-1)/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*weight/dimensionsArray[layer-1]+25+((height-50)/dimensionsArray[layer-1])/2)
        pop();
      }
    }
  }
}

function setInputs()
{
  for(let row = 0; row < dimensionsArray[0]; row++)
  {
    valuesArray[0][row] = int(random(0,2)); // only picks 0-1
  }
}

function drawnInput()
{
  //detect drawing and set pixels that have been drawn on equal to 1
  let pixelWidth = width/xPixels;
  let pixelHeight = height/yPixels;
  if(mouseIsPressed === true && mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height)
  {
    let x = floor(mouseX/pixelWidth);
    let y = floor(mouseY/pixelHeight);
    pixelsArray[x][y] = 1;
    print(x);
    print(y);
  }
  //pixelsArray[5][5] = 1;
  //pixelsArray[4][5] = 0;
  let counter = 0;
  for(y = 0; y < yPixels; y++)
  {
    for (x = 0; x < xPixels; x++)
    {
      if(pixelsArray[x][y] == 1) counter++;
      
    }
  }
  print(counter);
  // display all the pixels with the pixels that have been drawn to as black
  for(y = 0; y < yPixels; y++)
  {
    for (x = 0; x < xPixels; x++)
    {
      push();
      if(pixelsArray[x][y] == 1) fill(0,0,0);
      else fill(255,255,255);
      rect(x*pixelWidth,y*pixelHeight,pixelWidth,pixelHeight);
      pop();      
    }
  }
}

function setPixels()
{
  for(y = 0; y < yPixels; y++)
  {
    for (x = 0; x < xPixels; x++)
    {
      pixelsArray[x][y] = 0;
      
    }
  }
  
}


function setWeights()
{
  for(let layer = 0; layer < numberOfLayers; layer++)
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      //print(layer+1 + '  ' + row);
      //biasArray[layer+1][row] = float(random(-1,1));
      for(let weight = 0; weight < dimensionsArray[layer+1]; weight++)
      {
        weightsArray[layer][row][weight] = float(random(-1,1));
        //biasArray[layer+1][row] = float(random(-1,1));

      }
    }
  }
  for(let layer = 1; layer < numberOfLayers; layer++)
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      //print(layer+1 + '  ' + row);
      biasArray[layer][row] = float(random(-0.01,0.01));
      //print(biasArray[layer][row]);
    }
  }
}

function setArrays()
{
  for(layer = 0; layer < numberOfLayers; layer++)
  {
    valuesArray[layer] = [];
    weightsArray[layer] = [];
    biasArray[layer] = [];
    slopeArray[layer] = [];
    biasSlopeArray[layer] = [];
    for(row = 0; row < dimensionsArray[layer]; row++)
    {
      weightsArray[layer][row] = [];
      slopeArray[layer][row] = [];
    }
  }
  for(i = 0; i < xPixels; i++)
  {
    pixelsArray[i] = [];
  }
}