/***************************************************
this is a fully connected forward fed network that can be adjusted and trained. It is currently set to learn to recognize handwritten single-digit numbers and there is a pre-trained network that can be tested.

DRAW LARGE when testing the network!!! it should fill most of the screen but not all of it since that is what the network was trained on.
***************************************************/

//////////////////////////////////////////////
// adjust the values in this section to modify the network and how it trains as well as some other stuff, the program has to be reset for modifications to occur.

//hyperparameters
let dimensionsArray = [784,20,15,10,10]; // number of nodes in each layer, 784 input nodes in a 28 x 28 image inputs and outputs must not be changed if it is training on images.
let minibatchSize = 5; // sets the number of samples in a given minibatch, 
let loopsPerDisplay = 10; // loopsPerDisplay * minibatchSize = the number of images trained on per display, the larger this is the more taxing it will be to run, this will also change the time between displays
let lambda = 0.00001; // this adjusts the strength of L2 regularization, a higher value means weights will be pulled towards 0 quicker.
let learningRate = 0.05; // adjusts the learning rate of the network, a high value may prevent the network from converging, use a minibatch larger than one with a large learning rate.

// other stuff
let widthOfDisplay = 800; // adjust this to fit screen, usually 500 for laptops and 800 for desktopsl.
let usePreTrainedNetwork = true; // if this is set to false the network will start training from scratch. The network will be deleted once the program stops running
let brushSize = 1;
//////////////////////////////////////////////


// don't change these
let numberOfLayers;
let valuesArray = []; //2 dimensional layer, row
let backPropValuesArray = [];
let weightsArray = []; // 3 dimensional layer, row, weight
let biasArray = []; // same as valuesArray
let slopeArray = [] // same as weightsArray
let biasSlopeArray = []; // same as biasArray
let expectedValue = 7;
let expectedArray = [];// sample *up to minibatchSize*, inputs for that sample. Should contain all of the inputs in a minibatch
let minibatchInputs = [];
let derivativeWeights = [];
let derivativeBias = [];
let derivativeNodes = [];
let e = 2.7182818284;
let cost;
let pixelsArray = [];
let internalClock = 0;
let xPixels = 28;
let yPixels = 28;
let oldMouseX = 0;
let oldMouseY = 0;
let echo = false;
let epoch = 0;
let imagesTrainedNum = 0;
let maxSlope = 20;
let testSetSize = 5000 - 8;
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
let datasetImageNum = 60000;
let trainingDataIndex = 0;
let trainingDataMax = 55000;
let currentlyDrawing = false;
let currentlyTraining = false;
let waitForMemory = false;
let penDown = false;
let button1;
let button2;
let button3;
let button4;
let button5;
let button6;
let button7;
let button8;

function preload()
{ // load in the saved networks here
  file = loadJSON('NeuralNet1 (17).json');
  trainingData1 = loadBytes('train-images-updated (1).bin');
  trainingData2 = loadBytes('train-images-updated (2).bin');
  trainingData3 = loadBytes('train-images-updated (3).bin');
  //trainingData1 = loadBytes('testDataset1.bin');
  //trainingData2 = loadBytes('testDataset2.bin');
  //trainingData3 = loadBytes('testDataset3.bin');
  trainingLabels = loadBytes('train-labels (1).bin');

}


function setup() 
{
  noLoop();
  unzipFile();
  if (usePreTrainedNetwork === true)
  dimensionsArray = file.dimensions;
  let index = 0;
  numberOfLayers = 0;
  while(dimensionsArray[index] > 0)
  {
    numberOfLayers++;
    index++;
  }
  setArrays();
  createCanvas(widthOfDisplay, widthOfDisplay);
  setInputs();
  setWeights();
  setPixels();
  if (usePreTrainedNetwork === true)
  {
    weightsArray = file.weights;
    biasArray = file.bias;  
  }
  setNewInputs();
  background(220);
  propogate();
  drawNet();
  loop();
  
  
  
  button1 = createButton('drawImage');
  button1.position(20, 0);
  button1.mousePressed(drawOnScreen);
  button2 = createButton('clearCanvas');
  button2.position(98, 0);
  button2.mousePressed(clearDraw);
  button3 = createButton('seeDataset');
  button3.position(201, 0);
  button3.mousePressed(seeDatasetImage);
  button8 = createButton('seeNetwork');
  button8.position(282, 0);
  button8.mousePressed(stopTraining);
  button4 = createButton('resumeTraining');
  button4.position(381, 0);
  button4.mousePressed(resumeTraining);
  button5 = createButton('stopTraining');
  button5.position(486, 0);
  button5.mousePressed(stopTraining);
  button6 = createButton('printAccuracy');
  button6.position(589, 0);
  button6.mousePressed(printAccuracy);
  button7 = createButton('saveNetwork');
  button7.position(684, 0);
  button7.mousePressed(saveNetwork);
}


function draw() 
{
  if(currentlyDrawing === false && currentlyTraining === true)
  {
      for(let i = 0; i < loopsPerDisplay; i++)
    {
      setNewInputs();
      background(220);
      backpropogate3();
    }
    
    drawNet();
    imagesTrainedNum++;
    
    if (imagesTrainedNum % 250 == 0)
    {
      print('images trained: ' + (imagesTrainedNum * loopsPerDisplay * minibatchSize) + '   epochs: ' + epoch);
      print(avgCost());
      internalClock = 0;
      currentlyTraining = false;
      waitForMemory = true;
    }
    if ((imagesTrainedNum * loopsPerDisplay * minibatchSize) % 55000 == 0)
    {
      epoch++;
      trainingDataIndex++;
    }
  }
  else if(currentlyDrawing === true)
  {
    drawnInput();
  }
  else if(waitForMemory === true && internalClock == 0)
  {
    waitForMemory = false;
    currentlyTraining = true;
  }
  internalClock = (internalClock + 1)%500;
}

function setNewInputs()
{
  for (let i = 0; i < minibatchSize; i++)
  {
    let jitterX = 0;
    let jitterY = 0;
    if(trainingDataIndex > 0 && trainingDataIndex < 60000)
    {
      let boundsList = getImageBounds();
      let lowerXBound = boundsList[0];
      let lowerYBound = boundsList[1];
      let upperXBound = boundsList[2];
      let upperYBound = boundsList[3];
      jitterX = int(random(-(27-upperXBound),lowerXBound));
      jitterY = int(random(-(27-lowerYBound),upperYBound));
    
    }
    for ( let a = 0; a < 28*28; a++)
    {
      minibatchInputs[i][a] = trainingData[a + trainingDataIndex*784 + (jitterY * xPixels) + jitterX]
    }
    if ( jitterY > 0)
    {
      for (let c = 0; c < jitterY; c++)
      {
        for(let x = 0; x < xPixels; x++)
        {
          minibatchInputs[i][(((yPixels - 1) - jitterY) + c) * xPixels + x] = 0;
        }
      }
    }
    else
    {
      for (let c = 0; c < abs(jitterY); c++)
      {
        for(let x = 0; x < xPixels; x++)
        {
          minibatchInputs[i][(c) * xPixels + x] = 0;
        }
      }
    }
      
    for(let b = 0; b < 10; b++)
    {
      if (trainingLabels.bytes[trainingDataIndex + 8] == b)
        expectedArray[i][b] = 1;
      else expectedArray[i][b] = 0;     
    }
    trainingDataIndex = (trainingDataIndex + 1)%trainingDataMax;
  }
}

function getImageBounds()
{
  let boundsList = [];
  let upperYBound = 0; // should be thought of as an image where x = 0, y = 0 is at the bottom left of the screen. However the bounds will return an x or y where x = 0, y = 0 is in the upper left
  let lowerYBound = 0;
  let lowerXBound = 0;
  let upperXBound = 0;
  for(let y = 0; y < yPixels; y++)
  {
    for (let x = 0; x < xPixels; x++)
    {
      if (trainingData[trainingDataIndex*784 + x + y*xPixels] > 0 && upperYBound == 0)
      {
        upperYBound = y;
      }
    }
  }
  
  for(let y = yPixels-1; y >= 0; y--)
  {
    for(let x = 0; x < xPixels; x++)
    {
      if (trainingData[trainingDataIndex*784 + x + y*xPixels] > 0 && lowerYBound == 0)
      {
        lowerYBound = y;
      }
    }
  }
  
  for(let x = 0; x < xPixels; x++)
  {
    for(let y = 0; y < yPixels; y++)
    {
      if (trainingData[trainingDataIndex*784 + x + y*xPixels] > 0 && lowerXBound == 0)
      {
        lowerXBound = x;
      }
    }
  }
  
  for(let x = xPixels-1; x >= 0; x--)
  {
    for (let y = 0; y < yPixels; y++)
    {
      if (trainingData[trainingDataIndex*784 + x + y*xPixels] > 0 && upperXBound == 0)
      {
        upperXBound = x;
      }
    }
  }
  boundsList[0] = lowerXBound;
  boundsList[1] = lowerYBound;
  boundsList[2] = upperXBound;
  boundsList[3] = upperYBound;
  return boundsList;
}

function keyPressed()
{
  echo = true;
  
  if(keyCode ==  32)
  {
    let dataPlaceHolder = trainingDataIndex;
    let dataMaxPlaceHolder = trainingDataMax;
    trainingDataMax = 60000;
    trainingDataIndex = 50000 + int(random(0,5000));
    let temp = minibatchSize;
    minibatchSize = 1;
    setNewInputs();
    minibatchSize = temp;
    cost = propogate(expectedArray, 0);
    trainingDataIndex--;
    print('expected value: ' + trainingLabels.bytes[trainingDataIndex + 8]);
    let largestValue = 0;
    let largestValueNum = 0;
    for(i = 0; i <= 9; i++)
    {
      if (largestValue < valuesArray[numberOfLayers - 1][i])
      {
        largestValue = valuesArray[numberOfLayers - 1][i];
        largestValueNum = i;
      }   
    }
    print('prediction: ' + largestValueNum);
    background(220);
    drawNet();  
    drawImage();
    trainingDataMax = dataMaxPlaceHolder;
    trainingDataIndex = dataPlaceHolder;
  }
  
  if(keyCode === ENTER)
  {
    let smallArray = [0,1,2,3,4];
    let myJSON = {dimensions: dimensionsArray, weights: weightsArray, bias: biasArray};  
    smallArray[0] = [1,2,3,4,5];
    //save(myJSON, 'NeuralNet.json');
    save(myJSON, 'NeuralNet1.json');
    print('saved');
    
  }
  
  if(key == 'd')
  {
    trainingDataIndex--;
    drawImage();
    trainingDataIndex++;
  }
  
  if(key == 'a')
  {
    currentlyTraining = false;
    background(220);
    drawNet();
    if (currentlyDrawing === true)
      currentlyDrawing = false;
  }
  
  if(key == 'r')
  {
    currentlyTraining = true;
    loop();
  }
  
  if (key == 't')
  {
    print('error: ' + avgCost().toFixed(6));
  }
  
  if (key == 'f')
  {
    oldMouseX = 0;
    oldMouseY = 0;
    setPixels();
    if(currentlyDrawing === false)
    {
      loop();
      currentlyDrawing = true;
      background(0);
    }
    else if(currentlyDrawing === true)
    {
      noLoop();
      currentlyDrawing = false;
      setPixels();
      background(220);
      drawNet();
    }
    
  }
  if (key == 'c')
  {
    setPixels();
    oldMouseX = 0;
    oldMouseY = 0;
    if(currentlyDrawing === true)
      loop();
  }
}

function clearDraw()
{
  setPixels();
  oldMouseX = 0;
  oldMouseY = 0;
  currentlyDrawing = true;
  if(currentlyDrawing === true)
  loop();
}
function drawOnScreen()
{
  oldMouseX = 0;
  oldMouseY = 0;
  setPixels();
  if(currentlyDrawing === false)
  {
    loop();
    currentlyDrawing = true;
    background(0);
  }
    
}
function printAccuracy()
{
  print('error: ' + avgCost().toFixed(6));
}
function resumeTraining()
{
  currentlyDrawing = false;
  currentlyTraining = true;
  background(220);
  drawNet();
  loop();
}
function stopTraining()
{
  currentlyTraining = false;
  background(220);
  drawNet();
  if (currentlyDrawing === true)
    currentlyDrawing = false;
}
function seeImage()
{
  trainingDataIndex--;
  drawImage();
  trainingDataIndex++;
}
function saveNetwork()
{
  let smallArray = [0,1,2,3,4];
  let myJSON = {dimensions: dimensionsArray, weights: weightsArray, bias: biasArray};  
  smallArray[0] = [1,2,3,4,5];
  //save(myJSON, 'NeuralNet1.json');
  save(myJSON, 'NeuralNet.json');
  print('saved');
}
function seeDatasetImage()
{
  currentlyDrawing = false;
  currentlyTraining = false;
  penDown = false;
  let dataPlaceHolder = trainingDataIndex;
  let dataMaxPlaceHolder = trainingDataMax;
  trainingDataMax = 60000;
  trainingDataIndex = 55000 + int(random(0,4950));
  let temp = minibatchSize;
  minibatchSize = 1;
  setNewInputs();
  minibatchSize = temp;
  cost = propogate(expectedArray, 0);
  trainingDataIndex--;
  print('expected value: ' + trainingLabels.bytes[trainingDataIndex + 8]);
  let largestValue = 0;
  let largestValueNum = 0;
  for(i = 0; i <= 9; i++)
  {
    if (largestValue < valuesArray[numberOfLayers - 1][i])
    {
      largestValue = valuesArray[numberOfLayers - 1][i];
      largestValueNum = i;
    }   
  }
  print('prediction: ' + largestValueNum);
  background(220);
  drawNet();  
  drawImage();
  trainingDataMax = dataMaxPlaceHolder;
  trainingDataIndex = dataPlaceHolder;
}



function avgCost()
{
  //if(!minibatchSize || minibatchSize <= 0) minibatchSize = 1;
  let largestValue = 0;
  let largestValueNum = 0;
  let AvgCost = 0;
  let avgAccuracy = 0;
  let dataPlaceHolder = trainingDataIndex;
  let dataMaxPlaceHolder = trainingDataMax;
  trainingDataMax = 60000 - 8;
  trainingDataIndex = 6000;
  for(let i = 0; i < floor(testSetSize / minibatchSize); i++)
  {
    setNewInputs();
    for(let batchIndex = 0; batchIndex < minibatchSize; batchIndex++)
    {
      
      AvgCost += propogate(expectedArray, batchIndex)
      
      largestValue = 0;
      for(let a = 0; a <= 9; a++) // find the number that the network chose, the biggest number.
      {
        if (largestValue < valuesArray[numberOfLayers - 1][a])
        {
          largestValue = valuesArray[numberOfLayers - 1][a];
          largestValueNum = a;
        }   
      }
      if (largestValueNum == trainingLabels.bytes[(trainingDataIndex - (minibatchSize)) + batchIndex + 8])
      {
        avgAccuracy += 100;
      }
    }
  }
  trainingdataMax = dataMaxPlaceHolder;
  trainingDataIndex = dataPlaceHolder;
  avgAccuracy /= testSetSize;
  print('avgAccuracy: ' + avgAccuracy.toFixed(3) + '%')
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
        // The weight has the same dimensions as the next layer 
      }
      valuesArray[layer+1][weight] += biasArray[layer+1][weight];
      /////////////////
      //valuesArray[layer+1][weight] = 1 / (1 + (pow(e,-valuesArray[layer+1][weight]))); //sigmoid
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight] * 0.25 + 0.5); // ReLu
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight]); // Regular ReLu
      //valuesArray[layer+1][weight] = max(0,valuesArray[layer+1][weight]/2 + 0.01); // Regular ReLu small slope
      //valuesArray[layer+1][weight] = valuesArray[layer+1][weight]; // no ReLu
      valuesArray[layer+1][weight] = max(valuesArray[layer+1][weight]/10,valuesArray[layer+1][weight]); // leaky ReLU, best
      /////////////////
      // once the weight array progresses, all the values and weights have been added so the function can be applied before moving on to the next weight.
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
  for(let sampleIndex = 0; sampleIndex < minibatchSize; sampleIndex++)
  {
    propogate(expectedArray, sampleIndex);
    for(let row = 0; row < dimensionsArray[numberOfLayers-1]; row++)
    {
      // find the derivative of the last layer's nodes. 2 * 'expected value - actual value'
      derivativeNodes[numberOfLayers-1][row] = 2 * (valuesArray[numberOfLayers-1][row] - expectedArray[sampleIndex][row]);
      if(valuesArray[numberOfLayers-1][row] <= 0)
      {
        derivativeNodes[numberOfLayers-1][row] /= 10;
      }
      //derivativeNodes[numberOfLayers-1][row] = min(2,max(-2,derivativeNodes[numberOfLayers-1][row]));
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
          // find the derivative of each weight with respect to the output. nodeValue * derivative, or slope, of the target node.
          derivativeWeights[layer][row][weight] += valuesArray[layer][row] * derivativeNodes[layer+1][weight];
          // use derivativeNodes layer+1 weight to get the derivative of the target layer
          // find the derivative of the node with respect to the output. weightValue * derivative of the target node.
          derivativeNodes[layer][row] += weightsArray[layer][row][weight] * derivativeNodes[layer+1][weight];
        }
        // derive node
        // find the derivative of the bias with respect to the output. It is euqal to derivative, or slope, of the current node. Can be done on last layer as well
        if(valuesArray[layer][row] <= 0) 
        {
          derivativeNodes[layer][row] /= 10;
          derivativeBias[layer][row] += derivativeNodes[layer][row];
          //derivativeBias[layer][row] -= 0.00001;
        }
        else
        {
          //derivativeNodes[layer][row] = min(2,max(-2,derivativeNodes[layer][row]));
          derivativeBias[layer][row] += derivativeNodes[layer][row];
        }
      }
    }
    
  }

  
  for(let layer = numberOfLayers-2; layer >= 0; layer--) // starts on the second to last layer
  {
    for(let weight = 0; weight < dimensionsArray[layer+1]; weight++) // the number of weights in each node is equal to the number of nodes in the next layer
    {
      derivativeBias[layer+1][weight] /= minibatchSize;
      if (derivativeBias[layer+1][weight] < 10 && derivativeBias[layer+1][weight] > -10) {}
        else derivativeBias[layer+1][weight] = 0;
      biasArray[layer+1][weight] = biasArray[layer+1][weight] + ((derivativeBias[layer+1][weight] * (-1)) * learningRate);
      derivativeBias[layer+1][weight] = 0;
      for(let row = 0; row < dimensionsArray[layer]; row++)
      {
        derivativeWeights[layer][row][weight] /= minibatchSize;
        // adjust using L2 regularization
        derivativeWeights[layer][row][weight] += lambda * (2 * weightsArray[layer][row][weight]);
        derivativeWeights[layer][row][weight] = min(1, max(-1,derivativeWeights[layer][row][weight]));
        if(derivativeWeights[layer][row][weight] < 10 && derivativeWeights[layer][row][weight] > -10) {}
        else derivativeWeights[layer][row][weight] = 0;
        weightsArray[layer][row][weight] = weightsArray[layer][row][weight] + ((derivativeWeights[layer][row][weight] * (-1)) * learningRate);
        derivativeWeights[layer][row][weight] = 0;
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
      text(valuesArray[layer][row].toFixed(3), (width-50)*layer/numberOfLayers+25+((width-50)/numberOfLayers)/2, (height-50)*row/dimensionsArray[layer]+25+((height-50)/dimensionsArray[layer])/2,50)
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

  if(penDown === false)
  {
    penDown = true;
  }
  else if(penDown === true && mouseIsPressed === false)
  {
    penDown = false;
    oldMouseX = 0;
    oldMouseY = 0;
  }
  else
  {
    if(mouseIsPressed === true && mouseX > 0 && mouseX < width && mouseY > 20 && mouseY < height)
    {
      if(oldMouseX == 0 && oldMouseY == 0)
      {
        oldMouseX = mouseX;
        oldMouseY = mouseY;
      }
      for(let index = 0; index <= 5; index++)
      {
        let currentX = ((mouseX - oldMouseX) / 5) * index + oldMouseX;
        let currentY = ((mouseY - oldMouseY) / 5) * index + oldMouseY;
        let pixelsInputArray = [];
        let x = floor(currentX/pixelWidth);
        let y = floor(currentY/pixelHeight);
        for(let xIndex = -brushSize; xIndex <= brushSize; xIndex++)
        {
          for(let yIndex = -brushSize; yIndex <= brushSize; yIndex++)
          {
            if(x + xIndex >= 0 && x + xIndex < 28 && y + yIndex >= 0 && y + yIndex < 28)
            {
              let distanceX = abs(currentX - ((x + xIndex) * pixelWidth));
              let distanceY = abs(currentY - ((y + yIndex) * pixelHeight));
              let distance = sqrt((distanceX * distanceX) + (distanceY * distanceY))
              let pixelBrightness = (min(1.25, (brushSize * 1) / ((distance*distance) / (pixelWidth * 15))) - 0.25) * 255;
              if (pixelsArray[x + xIndex][y + yIndex] < pixelBrightness)
                pixelsArray[x + xIndex][y + yIndex] = pixelBrightness;
            }
          }
        }
      }
    }
    oldMouseX = mouseX;
    oldMouseY = mouseY;
  }

  // display all the pixels with the pixels that have been drawn to as white
  for(y = 0; y < yPixels; y++)
  {
    for (x = 0; x < xPixels; x++)
    {
      push();
      minibatchInputs[0][x + y * 28] = pixelsArray[x][y] / 255;
      if(pixelsArray[x][y] >= 0)
      fill(pixelsArray[x][y]);
      rect(x*pixelWidth,y*pixelHeight,pixelWidth,pixelHeight);
      pop();      
    }
  }
  
  if (internalClock%50 == 0)
  {
    propogate();
    let largestValue = 0;
    let largestValueNum = 0;
    let totalValue = 0;
    for(let i = 0; i <= 9; i++)
    {
      totalValue += abs(valuesArray[numberOfLayers - 1][i]);
      if (largestValue < valuesArray[numberOfLayers - 1][i])
      {
        largestValue = valuesArray[numberOfLayers - 1][i];
        largestValueNum = i;
      }   
    }
    let confidence = (valuesArray[numberOfLayers - 1][largestValueNum] / totalValue) * 100;
    print('prediction: ' + largestValueNum + '   confidence: ' + confidence.toFixed(3) + '%');    
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
      for(let weight = 0; weight < dimensionsArray[layer+1]; weight++)
      {
        weightsArray[layer][row][weight] = float(random(-1.0,1.0));
      }
    }
  }
  for(let layer = 1; layer < numberOfLayers; layer++)
  {
    for(let row = 0; row < dimensionsArray[layer]; row++)
    {
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

function drawImage()
{
  let jitterX = 0;
  let jitterY = 0;
  if(trainingDataIndex > 0 && trainingDataIndex < 60000)
  {
    let boundsList = getImageBounds();
    let lowerXBound = boundsList[0];
    let lowerYBound = boundsList[1];
    let upperXBound = boundsList[2];
    let upperYBound = boundsList[3];
    
    jitterX = int(random(-(27-upperXBound),lowerXBound));
    jitterY = int(random(-(27-lowerYBound),upperYBound));
    
  }
  for(a = 0; a < 28; a++)
  {
    for(b = 0; b < 28; b++)
    {
      let colorFill = trainingData[a*28 + b + trainingDataIndex * 784 + (jitterY * xPixels) + jitterX];
      push();
      fill(colorFill * 255);
      //noStroke();
      rect(width/28 * b, height/28 * a, width/28 + 1, height/28 + 1);
      pop();

    }
    if ( jitterY > 0)
    {
      for (let a = 0; a <= jitterY; a++)
      {
        for(let b = 0; b < xPixels; b++)
        {
          push();
          fill(0);
          //noStroke();
          rect(width/28 * b, height/28 * (27-jitterY + a), width/28 + 1, height/28 + 1);
          pop();

        }
      }
    }
    else
    {
      for (let a = 0; a < abs(jitterY); a++)
      {
        for(let b = 0; b < xPixels; b++)
        {
          push();
          fill(0);
          //noStroke();
          rect(width/28 * b, height/28 * a, width/28 + 1, height/28 + 1);
          pop();
        }
      }
    }
    
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

function unzipFile()
{
  // the MNIST dataset with all of the images was too big to be loaded in raw so I compressed it and am undoing my compression here
  let trainingDataCombined = [];
  let a = 0;
  let b = 0;

  for(b = 0; b < trainingData1.bytes.length; b++)
  {
    trainingDataCombined[a] = trainingData1.bytes[b]; 
    a++;

  }
  for(b = 0; b < trainingData2.bytes.length; b++)
  {
    trainingDataCombined[a] = trainingData2.bytes[b]; 
    a++;
  }
  for(b = 0; b < trainingData3.bytes.length; b++)
  {
    trainingDataCombined[a] = trainingData3.bytes[b]; 
    a++;
  }
  
  
  let trainingSize = 0;
  for(let i = 0; i < trainingDataCombined.length; i++)
  {      
    if(trainingDataCombined[i] > 0)
    {
      trainingData[trainingSize] = trainingDataCombined[i] / 255;
      trainingSize++;
    }
    else
    {
      trainingSize--;
      for(let zerosIndex = 0; zerosIndex < trainingDataCombined[i-1]; zerosIndex++)
      {
        trainingData[trainingSize + zerosIndex] = 0;
      }
      trainingSize += trainingDataCombined[i-1];
      if (trainingDataCombined[i-1] == 0)
        {
          trainingSize++;
          trainingData[trainingSize] = 0;
          trainingSize++;
        }
    }
  }
  
  trainingData1 = 0;
  trainingData2 = 0;
  trainingData3 = 0;
  trainingDataCombined = 0;

  
}