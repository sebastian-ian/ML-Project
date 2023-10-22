let numberOfLayers = 4;
let largestRow = 9;
let dimensionsArray = [1,7,7,7,7,7,7,1]; // number of nodes in each layer, 196 input nodes in a 14 x 14 image
let valuesArray = []; //2 dimensional layer, row
let backPropValuesArray = [];
let weightsArray = []; // 3 dimensional layer, row, weight
let biasArray = []; // same as valuesArray
let slopeArray = [] // same as weightsArray
let biasSlopeArray = []; // same as biasArray
let expectedValue = 7;
let expectedArray = [];// sample *up to minibatchSize*, inputs for that sample. Should contain all of the inputs in a minibatch
let minibatchInputs = []; // might use expectedArray instead
let minibatchSize = 50; // sets the number of samples in a given minibatch
let loopsPerDisplay = 10;
let derivativeWeights = [];
let derivativeBias = [];
let derivativeNodes = [];
let e = 2.7182818284;
let cost;
//let learningRate = 0.000005; // possibly 1.1 with delta weight 0.000001 and minibatchSize 10 or 0.1 with deltaweight 0.01 and mini 10
//let learningRate = 0.0005 * 20;
let learningRate = 0.0005;
let deltaWeight = 0.00001; // for backprop 2 used for finite differences
//let lambda = 0.01;
let lambda = 0.001 * 0.01;
let pixelsArray = [];
let xPixels = 16;
let yPixels = 16;
let echo = false;
let epoch = 0;
let maxSlope = 20;
//let testSetSize = 100000;
let testSetSize = 10000;
let input1;
let input2;
let input3;
let input4;
let file;
let trainingData1;
let trainingData2;
let trainingData3;
let trainingData = [];
let trainingLabels; // always ignore first 16 bytes of training labels
let datasetSize = 47040000;
let usePreTrainedNetwork = false;


function setup() {
  if (usePreTrainedNetwork === true)
  dimensionsArray = file.dimensions;
  //print(file.dimensions);
  /*if(loadNetwork === true)
  {
    dimensionsArray = 
  }*/
  //print(file);
  let index = 0;
  numberOfLayers = 0;
  while(dimensionsArray[index] > 0)
  {
    numberOfLayers++;
    index++;
  }
  setArrays();
  createCanvas(800, 800);
  setInputs();
  setWeights();
  //frameRate(1);
  setPixels();
  if (usePreTrainedNetwork === true)
  {
    weightsArray = file.weights;
    biasArray = file.bias;  
  }
  
  input1 = createInput('0.99');
  input1.position(0, 0);
  input1.size(100);
  input2 = createInput('0.99');
  input2.position(100, 0);
  input2.size(100);
  input3 = createInput('0.9');
  input3.position(200, 0);
  input3.size(100);
  input4 = createInput('0.9');
  input4.position(300, 0);
  input4.size(100);

}


function draw() 
{
    for(let i = 0; i < loopsPerDisplay; i++) // i is somehow connected to minibatchSize? Didn't initialize i
  {
    setNewInputs();

    background(220);
    //drawnInput();
    //cost = propogate(expectedArray, 0);
    //print(cost);
    backpropogate3();
    //print(weightsArray[3,0,0]);
    //noLoop();
    //print(valuesArray[2][1]);
  }
  drawNet();
  epoch++;
  if (epoch % 100 == 0)
  {
    print(epoch);
    let printAvgCost = avgCost();
    print('avgCost: ' + printAvgCost);
    if (printAvgCost < 0.00001) 
    {
      minibatchSize = 100;
      loopsPerDisplay = 250;
      learningRate = 0.0005;
      setMiniBatchArrays();
      // avgCost 0.00255306 at 20000 epochs 50 minibatchSize 100 loopsPerDisplay at 0 to 200 minibatchSize 25 loopsPerDisplay at 1000
    }
    //if (printAvgCost < 0.0005) keyPressed();
    //learningRate *= 1.5;
  }
  //if (epoch > 1000) keyPressed();

}

function setNewInputs()
{/*
  for(let i = 0; i < minibatchSize; i++)
  {
    expectedArray[i][0] = 0;
    let boolRandom = int(random(0,3));
    let newRandom = float(random(0.0,1.0));
    for(row = 0; row < 4; row++)
    {
      if(boolRandom == 1)
      minibatchInputs[i][row] = float(random(0.0,1.0));
      else minibatchInputs[i][row] = newRandom;
      expectedArray[i][0] += minibatchInputs[i][row];
    }
    expectedArray[i][0] /= 4;
    
  }*/
  for(let i = 0; i < minibatchSize; i++)
  {
    expectedArray[i][0] = float(random(0.0,10.0));    
    minibatchInputs[i][0] = expectedArray[i][0] * expectedArray[i][0];
  }
}

function keyPressed()
{
  echo = true;

  if (!input1.value() || !input2.value() || !input3.value() || !input4.value()) setNewInputs();
  else
  {
    expectedArray[0][0] = (float(input1.value()) + float(input2.value()) + float(input3.value()) + float(input4.value()))/4
    minibatchInputs[0][0] = float(input1.value());
    minibatchInputs[0][1] = float(input2.value());
    minibatchInputs[0][2] = float(input3.value());
    minibatchInputs[0][3] = float(input4.value());
    
  }
  setNewInputs();
  print('avgCost: ' + avgCost());
  cost = propogate(expectedArray, 0);
  print('expected value: ' + expectedArray[0]);
  print(valuesArray[numberOfLayers - 1][0]);
  background(220);
  drawNet();  
  noLoop();
  
  
  if(keyCode === ENTER)
  {
    let smallArray = [0,1,2,3,4];
    let myJSON = {dimensions: dimensionsArray, weights: weightsArray, bias: biasArray};  
    smallArray[0] = [1,2,3,4,5];
    //save(myJSON, 'NeuralNet.json');
    save(myJSON, 'NeuralNet.json');
    print('saved');
    
  }
}

function avgCost()
{
  if(!minibatchSize || minibatchSize <= 0) minibatchSize = 1;
  let AvgCost = 0;
  for(let i = 0; i < testSetSize / minibatchSize; i++)
  {
    setNewInputs();
    for(let batchIndex = 0; batchIndex < minibatchSize; batchIndex++)
    {
      AvgCost += propogate(expectedArray, batchIndex)
    }
  }
  AvgCost /= testSetSize;
  return AvgCost;
}

function propogate(expected, sampleIndex)
{
  if(!sampleIndex) sampleIndex = 0; 
  setBatchInputs(sampleIndex);
  for(let layer = 0; layer < numberOfLayers-1; layer++)
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
      //valuesArray[layer+1][weight] = 1 / (1 + (pow(e,-valuesArray[layer+1][weight]))); //sigmoid
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight] * 0.25 + 0.5); // ReLu
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight]); // Regular ReLu
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight]/2 + 0.01); // Regular ReLu small slope
      //valuesArray[layer+1][weight] = valuesArray[layer+1][weight]; // no ReLu
      valuesArray[layer+1][weight] = max(valuesArray[layer+1][weight]/10,valuesArray[layer+1][weight]); // leaky ReLU
      /////////////////
              //if(echo === true) print('relu' + ': ' + valuesArray[layer+1][weight]);
      // once the weight array progresses, all the values and weights have been added so the sigmoid can be applied before moving on to the next weight.
    }
  }
  if(!sampleIndex) sampleIndex = 0; 
  if(expected)
  {
    return calculateCost(expected, sampleIndex, valuesArray);
  }
}

function calculateCost(expected, sampleIndex, arrayValues)
{
  let currentCost = 0;
  for(row = 0; row < dimensionsArray[numberOfLayers-1]; row++) // compare the final layer to the expected values in the final row
  {
    currentCost += pow((arrayValues[numberOfLayers-1][row] - expected[sampleIndex][row]),2);
          //print(expected[sampleIndex][row]);
    //print(currentCost);
    // add up the differences between the final layer and the expected final layer squared to find the cost, inaccuracy, of the network
  }
  return currentCost;
}

function backpropogate3()
{
  /*
  find the derivative of a weight to see how it affects the cost function then increase it or decrease it proportionally.
  Use the chain rule
  */
  
  
  
  //propogate();
  // find the derivative of the last layer's nodes. 2 * 'expected value - actual value'
  
  // find the derivative of each weight with respect to the output. nodeValue * derivative, or slope, of the target node.
  // find the derivative of the bias with respect to the output. derivative, or slope, of the current node. Can be done on last layer as well
  // find the derivative of the node with respect to the output. weightValue * derivative of the target node.
  for(let sampleIndex = 0; sampleIndex < minibatchSize; sampleIndex++)
  {
    propogate(expectedArray, sampleIndex);
    for(let row = 0; row < dimensionsArray[numberOfLayers-1]; row++)
    {
      derivativeNodes[numberOfLayers-1][row] = 2 * (valuesArray[numberOfLayers-1][row] - expectedArray[sampleIndex][row]);
      if(valuesArray[numberOfLayers-1][row] <= 0)
      {
        derivativeNodes[numberOfLayers-1][row] /= 10;
      }
      derivativeNodes[numberOfLayers-1][row] = min(2,max(-2,derivativeNodes[numberOfLayers-1][row]));
      //derivativeBias[numberOfLayers-1][row] += derivativeNodes[numberOfLayers-1][row];
    }

    for(let layer = numberOfLayers-2; layer >= 0; layer--)
    {
      for(let row = 0; row < dimensionsArray[layer]; row++)
      {
        // derive bias
        derivativeNodes[layer][row] = 0; 
        for(let weight = 0; weight < dimensionsArray[layer+1]; weight++)
        {
          // derive weights
          //if (derivativeWeights[layer][row][weight] != 0 && sampleIndex == 0) print('adding twice');
          derivativeWeights[layer][row][weight] += valuesArray[layer][row] * derivativeNodes[layer+1][weight];
          //derivativeWeights[layer][row][weight] += min(20,max(-20,valuesArray[layer][row] * derivativeNodes[layer+1][weight]));
          // adjust using L2 regularization
              //derivativeWeights[layer][row][weight] += lambda * (2 * weightsArray[layer][row][weight]);

          // use derivativeNodes layer+1 weight to get the derivative of the target layer
          derivativeNodes[layer][row] += weightsArray[layer][row][weight] * derivativeNodes[layer+1][weight];
        }
        // derive node
        if(valuesArray[layer][row] <= 0) 
        {
          derivativeNodes[layer][row] /= 10;
          //derivativeBias[layer][row] -= 0.00001;
          //derivativeBias[layer][row] -= 0.001;
        }
        else
        {
          derivativeNodes[layer][row] = min(2,max(-2,derivativeNodes[layer][row]));
          derivativeBias[layer][row] += derivativeNodes[layer][row];
        }
      }
    }
    
  }

  
  for(let layer = numberOfLayers-2; layer >= 0; layer--) // starts on the second to last layer
  {
    for(let weight = 0; weight < dimensionsArray[layer+1]; weight++) // the number of weights in each node is equal to the number of nodes in the next layer
    {
      //print('before: ' + biasArray[layer+1][weight]);
      derivativeBias[layer+1][weight] /= minibatchSize;
      //if(derivativeBias < 0.01 && derivativeBias > -0.01) derivativeBias = -10.0;
      //derivativeBias[layer+1][weight] = min(maxSlope,derivativeBias[layer+1][weight]);
      biasArray[layer+1][weight] = biasArray[layer+1][weight] + ((derivativeBias[layer+1][weight] * (-1)) * learningRate);
      derivativeBias[layer+1][weight] = 0;
      //print('after: ' + biasArray[layer+1][weight]);
      //print(slopeArray[layer,row,weight]);
      for(let row = 0; row < dimensionsArray[layer]; row++)
      {
        derivativeWeights[layer][row][weight] /= minibatchSize;
        derivativeWeights[layer][row][weight] += lambda * (2 * weightsArray[layer][row][weight]);
        //derivativeWeights[layer][row][weight] = min(maxSlope,max(-maxSlope,derivativeWeights[layer][row][weight]));
        //if (derivativeWeights[layer][row][weight] > 20) print('too high: ' + derivativeWeights[layer][row][weight]);
        weightsArray[layer][row][weight] = weightsArray[layer][row][weight] + ((derivativeWeights[layer][row][weight] * (-1)) * learningRate);
        derivativeWeights[layer][row][weight] = 0;
        //print(weightsArray[layer][row][weight]);
      }
    }
  }
  
  
}

function drawNet()
{
  for(let layer = 0; layer < numberOfLayers; layer++)
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
      circle((width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,50);
      text(valuesArray[layer][row], (width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,50)
      for(let weight = 0; weight < dimensionsArray[layer+1]; weight++)
      {
        push();
        let colorValue = weightsArray[layer][row][weight];
        stroke((-colorValue)*200,colorValue*200,0); // colors the weight lines
        line((width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,(width-50)*(layer+1)/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*weight/dimensionsArray[layer+1]+25+((height-50)/dimensionsArray[layer+1])/2)
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
        weightsArray[layer][row][weight] = float(random(-1.0,1.0));
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
    }
  }
}

function setArrays()
{
  for(layer = 0; layer < numberOfLayers; layer++)
  {
    derivativeWeights[layer] = [];
    derivativeBias[layer] = [];
    derivativeNodes[layer] = [];
    valuesArray[layer] = [];
    backPropValuesArray[layer] = [];
    weightsArray[layer] = [];
    biasArray[layer] = [];
    slopeArray[layer] = [];
    biasSlopeArray[layer] = [];
    for(row = 0; row < dimensionsArray[layer]; row++)
    {
      derivativeBias[layer][row] = 0;
      derivativeWeights[layer][row] = [];
      biasSlopeArray[layer][row] = 0;
      weightsArray[layer][row] = [];
      slopeArray[layer][row] = [];
      for(weight = 0; weight < dimensionsArray[layer+1]; weight++)
      {
        derivativeWeights[layer][row][weight] = 0;
        slopeArray[layer][row][weight] = 0;
      }
    }
  }
  for(i = 0; i < xPixels; i++)
  {
    pixelsArray[i] = [];
  }
  for(i = 0; i < minibatchSize; i++)
  {
    minibatchInputs[i] = [];
    expectedArray[i] = [];
  }
}

function setMiniBatchArrays()
{
  for(i = 0; i < minibatchSize; i++)
  {
    minibatchInputs[i] = [];
    expectedArray[i] = [];
  }
}

function setBatchInputs(sampleIndex)
{
  for(row = 0; row < dimensionsArray[0]; row++)
  {
    valuesArray[0][row] = minibatchInputs[sampleIndex][row];
  }
}

