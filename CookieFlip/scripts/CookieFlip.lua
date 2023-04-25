--[[----------------------------------------------------------------------------

  Application Name:
  CookieFlip

  Summary:
  Training the system with binary classification to classify cookies as
  correctly oriented or flipped

  Description:
  Creating a training data set of cookie images, either correctly oriented or flipped.
  Training a classifier (SVM, kNN or Bayes) to do binary classification.
  Classifying a test set using the trained classifier.

  How to Run:
  Starting this sample is possible either by running the app (F5) or
  debugging (F7+F10). Setting breakpoint on the first row inside the 'main'
  function allows debugging step-by-step after 'Engine.OnStarted' event.
  Results can be seen in the image viewer on the DevicePage.
  Restarting the Sample may be necessary to show images after loading the webpage.
  To run this Sample a device with SICK Algorithm API and AppEngine >= V2.5.0 is
  required. For example SIM4000 with latest firmware. Alternatively the Emulator
  in AppStudio 2.3 or higher can be used.

  More Information:
  Tutorial "Algorithms - Machine Learning".

------------------------------------------------------------------------------]]
--Start of Global Scope---------------------------------------------------------

print('AppEngine Version: ' .. Engine.getVersion())

local DELAY = 1000 -- ms between visualization steps for demonstration purpose

--Classifier type
local CLASSIFIER_TYPE = 'SVM' -- Selecting "SVM", "kNN" or "Bayes"

-- Creating viewer
local viewer = View.create()

-- Setting up graphical overlay attributes
local passDecoration = View.ShapeDecoration.create():setLineColor(0, 255, 0) -- Green
passDecoration:setLineWidth(10):setFillColor(0, 255, 0, 40) -- Green, transparent

local failDecoration = View.ShapeDecoration.create():setLineColor(255, 0, 0) -- Red
failDecoration:setLineWidth(10):setFillColor(255, 0, 0, 40) -- Red, transparent

local textDecoration = View.TextDecoration.create():setPosition(20, 100)
textDecoration:setSize(90):setColor(0, 0, 255) -- Blue

-- Creating training DataSet
local trainingData = MachineLearning.DataSet.create('CLASSIFICATION')

-- Creating classifier, dending on set classifier type
local classifier

if CLASSIFIER_TYPE == 'SVM' then
  classifier = MachineLearning.SupportVectorMachine.create()
  classifier:setParameters(0, 'LINEAR')
  print('Classifier type: SVM')
elseif CLASSIFIER_TYPE == 'kNN' then
  classifier = MachineLearning.KNearestNeighbors.create()
  classifier:setParameters(4)
  print('Classifier type: kNN')
elseif CLASSIFIER_TYPE == 'Bayes' then
  classifier = MachineLearning.Bayes.create()
  print('Classifier type: Bayes')
else
  print('Classifier Type not specified correctly')
end

--End of Global Scope-----------------------------------------------------------

--Start of Function and Event Scope---------------------------------------------


--- Finding objects (blobs)
---@param img Image
---@return Image.PixelRegion[]
local function detectCookies(img)
  -- Threshold image
  local imgThr = img:threshold(0, 110)
  -- Removing holes and finding connected components
  imgThr = imgThr:fillHoles()
  local cookies = imgThr:findConnected(15000, 300000)
  return cookies
end

--- Extracting surface features
---@param img Image
---@param cookies Image.PixelRegion
---@return Matrix
local function computeCookieFeatures(img, cookies)
  -- Creating magnitude image once and for all
  local magIm = img:sobelMagnitude()

  -- Creating the output feature matrix
  local features = Matrix.create(#cookies, 4)
  for c = 1, #cookies do
    -- Using central part of cookie only
    local cookie = cookies[c]:erode(21)
    -- Selecting what features to use
    local min, max, avg, std = cookie:getStatistics(magIm)
    features:setRow(c - 1, Matrix.createFromVector({min, max, avg, std}, 1, 4))
  end

  return features
end

---@param img Image
---@param label string
local function addTrainingSamples(img, label)
  local cookies = detectCookies(img)
  if #cookies == 0 then
    return
  end
  -- Adding sampels from original image
  local features = computeCookieFeatures(img, cookies)
  trainingData:append(features, label)
end

-- Creating training data
local function createTrainingData()
  -- Negatives (not flipped)
  for i = 1, 2 do
    local img = Image.load('resources/Train/negatives_' .. tostring(i) .. '.png')
    addTrainingSamples(img, 1) -- 1 = not flipped
    viewer:clear()
    viewer:addImage(img)
    viewer:addText('Train negatives', textDecoration)
    viewer:present()
    Script.sleep(DELAY)
  end

  -- Positives (flipped)
  for i = 1, 2 do
    local img = Image.load('resources/Train/positives_' .. tostring(i) .. '.png')
    addTrainingSamples(img, 2) -- 2 = flipped
    viewer:clear()
    viewer:addImage(img)
    viewer:addText('Train positives', textDecoration)
    viewer:present()
    Script.sleep(DELAY)
  end
  print(trainingData:toString())
end

-- Training classifier
local function trainClassifier()
  local success = classifier:train(trainingData)
  if not success then
    print('Training failed')
    return
  end
  local accuracy, confusionMatrix = classifier:getAccuracy(trainingData)
  print( 'Trained classifier with accuracy: ' .. tostring(accuracy * 100) .. ' %' )
  print('Confusion matrix = \n' .. confusionMatrix:toString())
end

-- Loading and classifying test images
local function classify()
  for i = 1, 2 do
    local currentImage = Image.load('resources/Test/mix_' .. tostring(i) .. '.png')
    viewer:clear()
    viewer:addImage(currentImage)

    local tic = DateTime.getTimestamp()
    local cookies = detectCookies(currentImage)
    if #cookies == 0 then
      print('No cookies in image')
      return
    end
    if not classifier then
      print('Classifier was not trained')
      return
    end

    local features = computeCookieFeatures(currentImage, cookies)
    local labels = classifier:predict(features)
    local toc = DateTime.getTimestamp()
    local procTime = toc - tic
    print('Processing time = ' .. procTime .. ' ms')

    local failCount = 0
    -- Drawing results
    for c = 1, #cookies do
      local box = cookies[c]:getBoundingBoxOriented(currentImage)
      local label = type(labels) == 'table' and labels[c] or labels
      if label == 2 then
        viewer:addShape(box, passDecoration)
      else
        viewer:addShape(box, failDecoration)
        failCount = failCount + 1
      end
    end
    viewer:addText('Classify', textDecoration)
    viewer:present()
    print(failCount .. ' out of ' .. #cookies .. ' cookies are flipped')

    Script.sleep(DELAY)
  end
end

local function main()
  createTrainingData()
  trainClassifier()
  classify()
  print('App finished.')
end
--The following registration is part of the global scope which runs once after startup
--Registration of the 'main' function to the 'Engine.OnStarted' event
Script.register('Engine.OnStarted', main)

--End of Function and Event Scope--------------------------------------------------
