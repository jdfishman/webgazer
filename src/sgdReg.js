'use strict';
(function(window) {

    window.webgazer = window.webgazer || {};
    webgazer.reg = webgazer.reg || {};
    webgazer.mat = webgazer.mat || {};
    webgazer.util = webgazer.util || {};
    webgazer.params = webgazer.params || {};

    var ridgeParameter = Math.pow(10,-5);
    var resizeWidth = 10;
    var resizeHeight = 6;
    var dataWindow = 700;
    var trailDataWindow = 10;

    //Parameters
    var trainedCoefficientsX;
    var trainedCoefficientsY;

    /**
     * Performs ridge regression, according to the Weka code.
     * @param {Array} y - corresponds to screen coordinates (either x or y) for each of n click events
     * @param {Array.<Array.<Number>>} X - corresponds to gray pixel features (120 pixels for both eyes) for each of n clicks
     * @param {Array} k - ridge parameter
     * @return{Array} regression coefficients
     */
    function ridge(y, X, k){
        var nc = X[0].length;
        var m_Coefficients = new Array(nc);
        var xt = webgazer.mat.transpose(X);
        var solution = new Array();
        var success = true;
        do{
            var ss = webgazer.mat.mult(xt,X);
            // Set ridge regression adjustment
            for (var i = 0; i < nc; i++) {
                ss[i][i] = ss[i][i] + k;
            }

            // Carry out the regression
            var bb = webgazer.mat.mult(xt,y);
            for(var i = 0; i < nc; i++) {
                m_Coefficients[i] = bb[i][0];
            }
            try{
                var n = (m_Coefficients.length !== 0 ? m_Coefficients.length/m_Coefficients.length: 0);
                if (m_Coefficients.length*n !== m_Coefficients.length){
                    console.log('Array length must be a multiple of m')
                }
                solution = (ss.length === ss[0].length ? (numeric.LUsolve(numeric.LU(ss,true),bb)) : (webgazer.mat.QRDecomposition(ss,bb)));

                for (var i = 0; i < nc; i++){
                    m_Coefficients[i] = solution[i];
                }
                success = true;
            }
            catch (ex){
                k *= 10;
                console.log(ex);
                success = false;
            }
        } while (!success);
        return m_Coefficients;
    }
    
    /**
     * Compute eyes size as gray histogram
     * @param {Object} eyes - The eyes where looking for gray histogram
     * @returns {Array.<T>} The eyes gray level histogram
     */
    function getEyeFeats(eyes) {
        var resizedLeft = webgazer.util.resizeEye(eyes.left, resizeWidth, resizeHeight);
        var resizedright = webgazer.util.resizeEye(eyes.right, resizeWidth, resizeHeight);

        var leftGray = webgazer.util.grayscale(resizedLeft.data, resizedLeft.width, resizedLeft.height);
        var rightGray = webgazer.util.grayscale(resizedright.data, resizedright.width, resizedright.height);

        var histLeft = [];
        webgazer.util.equalizeHistogram(leftGray, 5, histLeft);
        var histRight = [];
        webgazer.util.equalizeHistogram(rightGray, 5, histRight);

        var leftGrayArray = Array.prototype.slice.call(histLeft);
        var rightGrayArray = Array.prototype.slice.call(histRight);

        return leftGrayArray.concat(rightGrayArray);
    }

    //TODO: still usefull ???
    /**
     *
     * @returns {Number}
     */
    function getCurrentFixationIndex() {
        var index = 0;
        var recentX = this.screenXTrailArray.get(0);
        var recentY = this.screenYTrailArray.get(0);
        for (var i = this.screenXTrailArray.length - 1; i >= 0; i--) {
            var currX = this.screenXTrailArray.get(i);
            var currY = this.screenYTrailArray.get(i);
            var euclideanDistance = Math.sqrt(Math.pow((currX-recentX),2)+Math.pow((currY-recentY),2));
            if (euclideanDistance > 72){
                return i+1;
            }
        }
        return i;
    }

    /**
     * Constructor of RidgeReg object,
     * this object allow to perform ridge regression
     * @constructor
     */
    webgazer.reg.sgdReg = function() {
        this.screenXClicksArray = new webgazer.util.DataWindow(dataWindow);
        this.screenYClicksArray = new webgazer.util.DataWindow(dataWindow);
        this.eyeFeaturesClicks = new webgazer.util.DataWindow(dataWindow);

        //sets to one second worth of cursor trail
        this.trailTime = 1000;
        this.trailDataWindow = this.trailTime / webgazer.params.moveTickSize;
        this.screenXTrailArray = new webgazer.util.DataWindow(trailDataWindow);
        this.screenYTrailArray = new webgazer.util.DataWindow(trailDataWindow);
        this.eyeFeaturesTrail = new webgazer.util.DataWindow(trailDataWindow);
        this.trailTimes = new webgazer.util.DataWindow(trailDataWindow);

        this.dataClicks = new webgazer.util.DataWindow(dataWindow);
        this.dataTrail = new webgazer.util.DataWindow(dataWindow);

        //Load in trained weights
        trainedCoefficientsX = x_sgd_weights;
        trainedCoefficientsY = y_sgd_weights;

        /*trainedCoefficientsX = x_ridge_weights.map(function (x) {return x * screen.width});
        trainedCoefficientsY = y_ridge_weights.map(function (x) {return x * screen.height});
        //make unit vector
        var xMag = 0;
        for (i = 0; i < trainedCoefficientsX.length; i++) {
            xMag += trainedCoefficientsX[i]**2;
        }
        trainedMagX = Math.sqrt(xMag);
        var yMag = 0;
        for (i = 0; i < trainedCoefficientsY.length; i++) {
            yMag += trainedCoefficientsY[i]**2;
        }
        trainedMagY = Math.sqrt(yMag);*/

    };

    /**
     * Add given data from eyes
     * @param {Object} eyes - eyes where extract data to add
     * @param {Object} screenPos - The current screen point
     * @param {Object} type - The type of performed action
     */
    webgazer.reg.sgdReg.prototype.addData = function(eyes, screenPos, type) {
        if (!eyes) {
            return;
        }
        if (eyes.left.blink || eyes.right.blink) {
            return;
        }
        if (type === 'click') {
            this.screenXClicksArray.push([screenPos[0]]);
            this.screenYClicksArray.push([screenPos[1]]);

            this.eyeFeaturesClicks.push(getEyeFeats(eyes));
            this.dataClicks.push({'eyes':eyes, 'screenPos':screenPos, 'type':type});
        } else if (type === 'move') {
            this.screenXTrailArray.push([screenPos[0]]);
            this.screenYTrailArray.push([screenPos[1]]);

            this.eyeFeaturesTrail.push(getEyeFeats(eyes));
            this.trailTimes.push(performance.now());
            this.dataTrail.push({'eyes':eyes, 'screenPos':screenPos, 'type':type});
        }

        eyes.left.patch = Array.from(eyes.left.patch.data);
        eyes.right.patch = Array.from(eyes.right.patch.data);
    }

    /**
     * Try to predict coordinates from pupil data
     * after apply linear regression on data set
     * @param {Object} eyesObj - The current user eyes object
     * @returns {Object}
     */
    webgazer.reg.sgdReg.prototype.predict = function(eyesObj) {
        if (!eyesObj) {
            return null;
        }

        var coefficientsX = trainedCoefficientsX;
        var coefficientsY = trainedCoefficientsY;
        /*
        if (this.eyeFeaturesClicks.length != 0) {
            var acceptTime = performance.now() - this.trailTime;
            var trailX = [];
            var trailY = [];
            var trailFeat = [];
            for (var i = 0; i < this.trailDataWindow; i++) {
                if (this.trailTimes.get(i) > acceptTime) {
                    trailX.push(this.screenXTrailArray.get(i));
                    trailY.push(this.screenYTrailArray.get(i));
                    trailFeat.push(this.eyeFeaturesTrail.get(i));
                }
            }

            var screenXArray = this.screenXClicksArray.data.concat(trailX);
            var screenYArray = this.screenYClicksArray.data.concat(trailY);
            var eyeFeatures = this.eyeFeaturesClicks.data.concat(trailFeat);

            var newCoefficientsX = ridge(screenXArray, eyeFeatures, ridgeParameter);
            newCoefficientsX.push(0); //bias term
            var newCoefficientsY = ridge(screenYArray, eyeFeatures, ridgeParameter);
            newCoefficientsY.push(0); //bias term

            new_weight = Math.exp(-5/screenXArray.length);
            training_weight = 1-new_weight;

            var xMag = 0;
            for (i = 0; i < newCoefficientsX.length; i++) {
                xMag += newCoefficientsX[i]**2;
            }
            xMag = Math.sqrt(xMag);
            var yMag = 0;
            for (i = 0; i < newCoefficientsY.length; i++) {
                yMag += newCoefficientsY[i]**2;
            }
            yMag = Math.sqrt(yMag);

            coefficientsX = trainedCoefficientsX.map(function(x) {return x * training_weight * xMag / trainedMagX}).map((a, i) => a + newCoefficientsX.map(function(x) {return x * new_weight} )[i]); 
            coefficientsY = trainedCoefficientsY.map(function(x) {return x * training_weight * yMag / trainedMagY}).map((a, i) => a + newCoefficientsY.map(function(x) {return x * new_weight} )[i]);
        }*/

        var eyeFeats = getEyeFeats(eyesObj);
        var featsMag = 0;
        for (i = 0; i < eyeFeats.length; i++) {
            featsMag += eyeFeats[i]**2;
        }
        featsMag = Math.sqrt(featsMag);
        normalizedEyeFeats = eyeFeats.map(function(x) {return x / featsMag;});

        normalizedEyeFeats.push(1); //bias term

        var predictedX = 0;
        for(var i=0; i< normalizedEyeFeats.length; i++){
            predictedX += normalizedEyeFeats[i] * coefficientsX[i];
        }
        var predictedY = 0;
        for(var i=0; i< normalizedEyeFeats.length; i++){
            predictedY += normalizedEyeFeats[i] * coefficientsY[i];
        }

        predictedX = Math.floor(predictedX*screen.width);
        predictedY = Math.floor(predictedY*screen.height);

        return {
            x: predictedX,
            y: predictedY
        };
    };


    /**
     * Add given data to current data set then,
     * replace current data member with given data
     * @param {Array.<Object>} data - The data to set
     */
    webgazer.reg.sgdReg.prototype.setData = function(data) {
        for (var i = 0; i < data.length; i++) {
            //TODO this is a kludge, needs to be fixed
            data[i].eyes.left.patch = new ImageData(new Uint8ClampedArray(data[i].eyes.left.patch), data[i].eyes.left.width, data[i].eyes.left.height);
            data[i].eyes.right.patch = new ImageData(new Uint8ClampedArray(data[i].eyes.right.patch), data[i].eyes.right.width, data[i].eyes.right.height);
            this.addData(data[i].eyes, data[i].screenPos, data[i].type);
        }
    };

    /**
     * Return the data
     * @returns {Array.<Object>|*}
     */
    webgazer.reg.sgdReg.prototype.getData = function() {
        return this.dataClicks.data.concat(this.dataTrail.data);
    }
    
    /**
     * The RidgeReg object name
     * @type {string}
     */
    webgazer.reg.sgdReg.prototype.name = 'ridge';
    
}(window));
