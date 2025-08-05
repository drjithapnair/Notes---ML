## 5. Random Forest

### Random Forest Ensemble Visualization

<div style="width: 100%; height: 600px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="randomForestCanvas" width="800" height="600"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Random Forest Ensemble</strong><br>
Click to add points:<br>
<button id="rfClassA" style="background: #2196F3; color: white; border: none; padding: 5px;">Class A (Blue)</button>
<button id="rfClassB" style="background: #f44336; color: white; border: none; padding: 5px;">Class B (Red)</button><br>
<label>Trees: <input type="range" id="numTrees" min="1" max="10" value="5"></label> <span id="treeCount">5</span><br>
<button id="trainForest" style="margin-top: 5px; padding: 5px;">Train Forest</button>
<button id="clearForest" style="margin-top: 5px; padding: 5px;">Clear</button><br>
<button id="showTree" style="margin-top: 5px; padding: 5px;">Show Tree <span id="currentTree">1</span></button><br>
<span id="forestInfo">Accuracy: 0%</span>
</div>
</div>

<script>
const rfCanvas = document.getElementById('randomForestCanvas');
const rfCtx = rfCanvas.getContext('2d');
let rfPoints = [];
let currentRfClass = 'A';
let forest = [];
let currentTreeIndex = 0;
let showingIndividualTree = false;

document.getElementById('rfClassA').addEventListener('click', () => {
    currentRfClass = 'A';
    document.getElementById('rfClassA').style.fontWeight = 'bold';
    document.getElementById('rfClassB').style.fontWeight = 'normal';
});

document.getElementById('rfClassB').addEventListener('click', () => {
    currentRfClass = 'B';
    document.getElementById('rfClassB').style.fontWeight = 'bold';
    document.getElementById('rfClassA').style.fontWeight = 'normal';
});

function bootstrapSample(points) {
    const sample = [];
    for(let i = 0; i < points.length; i++) {
        const randomIndex = Math.floor(Math.random() * points.length);
        sample.push(points[randomIndex]);
    }
    return sample;
}

function buildRandomTree(points) {
    // Simplified random tree building
    const splits = [];
    let regions = [{points: [...points], bounds: {x1: 0, y1: 0, x2: rfCanvas.width, y2: rfCanvas.height}}];
    
    for(let depth = 0; depth < 2; depth++) {
        let newRegions = [];
        
        for(let region of regions) {
            if(region.points.length < 2) {
                newRegions.push(region);
                continue;
            }
            
            // Random feature selection (x or y)
            const useX = Math.random() > 0.5;
            let bestSplit = null;
            let bestGain = 0;
            
            // Try random split positions
            for(let i = 0; i < 5; i++) {
                let splitPos;
                if(useX) {
                    splitPos = region.bounds.x1 + Math.random() * (region.bounds.x2 - region.bounds.x1);
                } else {
                    splitPos = region.bounds.y1 + Math.random() * (region.bounds.y2 - region.bounds.y1);
                }
                
                const left = region.points.filter(p => useX ? p.x < splitPos : p.y < splitPos);
                const right = region.points.filter(p => useX ? p.x >= splitPos : p.y >= splitPos);
                
                if(left.length === 0 || right.length === 0) continue;
                
                const entropy = calculateRfEntropy(region.points);
                const leftEntropy = calculateRfEntropy(left);
                const rightEntropy = calculateRfEntropy(right);
                
                const gain = entropy - (left.length / region.points.length) * leftEntropy 
                           - (right.length / region.points.length) * rightEntropy;
                
                if(gain > bestGain) {
                    bestGain = gain;
                    bestSplit = {
                        type: useX ? 'vertical' : 'horizontal',
                        position: splitPos,
                        left: left,
                        right: right
                    };
                }
            }
            
            if(bestSplit && bestGain > 0.1) {
                splits.push({
                    type: bestSplit.type,
                    position: bestSplit.position,
                    bounds: region.bounds
                });
                
                if(bestSplit.type === 'vertical') {
                    newRegions.push({
                        points: bestSplit.left,
                        bounds: {x1: region.bounds.x1, y1: region.bounds.y1, x2: bestSplit.position, y2: region.bounds.y2}
                    });
                    newRegions.push({
                        points: bestSplit.right,
                        bounds: {x1: bestSplit.position, y1: region.bounds.y1, x2: region.bounds.x2, y2: region.bounds.y2}
                    });
                } else {
                    newRegions.push({
                        points: bestSplit.left,
                        bounds: {x1: region.bounds.x1, y1: region.bounds.y1, x2: region.bounds.x2, y2: bestSplit.position}
                    });
                    newRegions.push({
                        points: bestSplit.right,
                        bounds: {x1: region.bounds.x1, y1: bestSplit.position, x2: region.bounds.x2, y2: region.bounds.y2}
                    });
                }
            } else {
                newRegions.push(region);
            }
        }
        
        regions = newRegions;
    }
    
    return {splits: splits, regions: regions};
}

function calculateRfEntropy(points) {
    if(points.length === 0) return 0;
    const classA = points.filter(p => p.class === 'A').length;
    const classB = points.filter(p => p.class === 'B').length;
    const total = points.length;
    
    if(classA === 0 || classB === 0) return 0;
    
    const pA = classA / total;
    const pB = classB / total;
    
    return -(pA * Math.log2(pA) + pB * Math.log2(pB));
}

function trainRandomForest() {
    const numTrees = parseInt(document.getElementById('numTrees').value);
    forest = [];
    
    for(let i = 0; i < numTrees; i++) {
        const bootstrapData = bootstrapSample(rfPoints);
        const tree = buildRandomTree(bootstrapData);
        forest.push(tree);
    }
    
    drawRandomForest();
}

function drawRandomForest() {
    rfCtx.clearRect(0, 0, rfCanvas.width, rfCanvas.height);
    
    // Draw grid
    rfCtx.strokeStyle = '#f0f0f0';
    rfCtx.lineWidth = 1;
    for(let i = 0; i <= rfCanvas.width; i += 50) {
        rfCtx.beginPath();
        rfCtx.moveTo(i, 0);
        rfCtx.lineTo(i, rfCanvas.height);
        rfCtx.stroke();
    }
    for(let i = 0; i <= rfCanvas.height; i += 50) {
        rfCtx.beginPath();
        rfCtx.moveTo(0, i);
        rfCtx.lineTo(rfCanvas.width, i);
        rfCtx.stroke();
    }
    
    if(forest.length > 0) {
        if(showingIndividualTree) {
            // Show individual tree
            const tree = forest[currentTreeIndex];
            rfCtx.strokeStyle = '#4CAF50';
            rfCtx.lineWidth = 2;
            
            tree.splits.forEach(split => {
                rfCtx.beginPath();
                if(split.type === 'vertical') {
                    rfCtx.moveTo(split.position, split.bounds.y1);
                    rfCtx.lineTo(split.position, split.bounds.y2);
                } else {
                    rfCtx.moveTo(split.bounds.x1, split.position);
                    rfCtx.lineTo(split.bounds.x2, split.position);
                }
                rfCtx.stroke();
            });
        } else {
            // Show ensemble prediction
            const gridSize = 10;
            for(let x = 0; x < rfCanvas.width; x += gridSize) {
                for(let y = 0; y < rfCanvas.height; y += gridSize) {
                    let votes = {A: 0, B: 0};
                    
                    // Get prediction from each tree
                    forest.forEach(tree => {
                        const prediction = predictWithTree(tree, {x: x, y: y});
                        votes[prediction]++;
                    });
                    
                    const finalPrediction = votes.A > votes.B ? 'A' : 'B';
                    const confidence = Math.max(votes.A, votes.B) / forest.length;
                    
                    rfCtx.fillStyle = finalPrediction === 'A' ? 
                        `rgba(33, 150, 243, ${confidence * 0.3})` : 
                        `rgba(244, 67, 54, ${confidence * 0.3})`;
                    rfCtx.fillRect(x, y, gridSize, gridSize);
                }
            }
        }
        
        // Calculate accuracy
        let correct = 0;
        rfPoints.forEach(point => {
            let votes = {A: 0, B: 0};
            forest.forEach(tree => {
                const prediction = predictWithTree(tree, point);
                votes[prediction]++;
            });
            const finalPrediction = votes.A > votes.B ? 'A' : 'B';
            if(finalPrediction === point.class) correct++;
        });
        
        const accuracy = rfPoints.length > 0 ? (correct / rfPoints.length * 100).toFixed(1) : 0;
        document.getElementById('forestInfo').textContent = `Accuracy: ${accuracy}%`;
    }
    
    // Draw points
    rfPoints.forEach(point => {
        rfCtx.fillStyle = point.class === 'A' ? '#2196F3' : '#f44336';
        rfCtx.beginPath();
        rfCtx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
        rfCtx.fill();
    });
    
    document.getElementById('treeCount').textContent = document.getElementById('numTrees').value;
}

function predictWithTree(tree, point) {
    // Find which region the point falls into
    for(let region of tree.regions) {
        if(point.x >= region.bounds.x1 && point.x < region.bounds.x2 &&
           point.y >= region.bounds.y1 && point.y < region.bounds.y2) {
            
            if(region.points.length === 0) return 'A';
            
            const classA = region.points.filter(p => p.class === 'A').length;
            const classB = region.points.filter(p => p.class === 'B').length;
            
            return classA > classB ? 'A' : 'B';
        }
    }
    return 'A';
}

rfCanvas.addEventListener('click', (e) => {
    const rect = rfCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    rfPoints.push({x, y, class: currentRfClass});
    drawRandomForest();
});

document.getElementById('trainForest').addEventListener('click', trainRandomForest);
document.getElementById('clearForest').addEventListener('click', () => {
    rfPoints = [];
    forest = [];
    drawRandomForest();
});

document.getElementById('showTree').addEventListener('click', () => {
    if(forest.length > 0) {
        showingIndividualTree = !showingIndividualTree;
        if(showingIndividualTree) {
            currentTreeIndex = (currentTreeIndex + 1) % forest.length;
            document.getElementById('currentTree').textContent = currentTreeIndex + 1;
            document.getElementById('showTree').textContent = `Show Ensemble`;
        } else {
            document.getElementById('showTree').textContent = `Show Tree ${currentTreeIndex + 1}`;
        }
        drawRandomForest();
    }
});

document.getElementById('numTrees').addEventListener('input', () => {
    if(forest.length > 0) trainRandomForest();
});

// Initialize with sample points
rfPoints = [
    {x: 150, y: 150, class: 'A'}, {x: 180, y: 120, class: 'A'}, {x: 120, y: 180, class: 'A'},
    {x: 200, y: 200, class: 'A'}, {x: 500, y: 400, class: 'B'}, {x: 530, y: 370, class: 'B'},
    {x: 470, y: 430, class: 'B'}, {x: 600, y: 350, class: 'B'}
];

drawRandomForest();
</script># Machine Learning Algorithms: Complete Guide

## 1. Linear Regression

### Interactive Visualization

<div style="width: 100%; height: 400px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="linearRegressionCanvas" width="800" height="400" style="cursor: crosshair;"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Linear Regression Demo</strong><br>
Click to add points. The red line shows the best fit.<br>
<span id="linearEquation">y = 0.00x + 0.00</span><br>
<span id="linearR2">R² = 0.00</span>
</div>
</div>

<script>
// Linear Regression Visualization
const canvas = document.getElementById('linearRegressionCanvas');
const ctx = canvas.getContext('2d');
let points = [];

// Initial sample points
points = [
    {x: 100, y: 300}, {x: 200, y: 250}, {x: 300, y: 200}, 
    {x: 400, y: 180}, {x: 500, y: 120}, {x: 600, y: 100}
];

function drawLinearRegression() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    for(let i = 0; i <= canvas.width; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }
    for(let i = 0; i <= canvas.height; i += 50) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
    }
    
    // Draw points
    ctx.fillStyle = '#2196F3';
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    if(points.length > 1) {
        // Calculate linear regression
        const n = points.length;
        const sumX = points.reduce((sum, p) => sum + p.x, 0);
        const sumY = points.reduce((sum, p) => sum + p.y, 0);
        const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
        const sumX2 = points.reduce((sum, p) => sum + p.x * p.x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Draw regression line
        ctx.strokeStyle = '#f44336';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, intercept);
        ctx.lineTo(canvas.width, slope * canvas.width + intercept);
        ctx.stroke();
        
        // Calculate R²
        const yMean = sumY / n;
        const ssRes = points.reduce((sum, p) => {
            const predicted = slope * p.x + intercept;
            return sum + Math.pow(p.y - predicted, 2);
        }, 0);
        const ssTot = points.reduce((sum, p) => sum + Math.pow(p.y - yMean, 2), 0);
        const r2 = 1 - (ssRes / ssTot);
        
        // Update equation display
        document.getElementById('linearEquation').textContent = 
            `y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`;
        document.getElementById('linearR2').textContent = 
            `R² = ${r2.toFixed(3)}`;
    }
}

canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    points.push({x, y});
    drawLinearRegression();
});

// Double click to clear
canvas.addEventListener('dblclick', () => {
    points = [];
    drawLinearRegression();
});

drawLinearRegression();
</script>

### Mathematical Foundation
Linear regression models the relationship between a dependent variable y and independent variables X using a linear equation:

**Simple Linear Regression:**
```
y = β₀ + β₁x + ε
```

**Multiple Linear Regression:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y = dependent variable (target)
- β₀ = y-intercept (bias term)
- β₁, β₂, ..., βₙ = coefficients (weights)
- x₁, x₂, ..., xₙ = independent variables (features)
- ε = error term

### Cost Function
The algorithm minimizes the Mean Squared Error (MSE):
```
MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

### How It Works
1. **Initialize** coefficients randomly
2. **Calculate predictions** using current coefficients
3. **Compute cost** using MSE
4. **Update coefficients** using gradient descent:
   ```
   β₁ = β₁ - α × (∂MSE/∂β₁)
   ```
5. **Repeat** until convergence

### Key Assumptions
- Linear relationship between variables
- Independence of residuals
- Homoscedasticity (constant variance)
- Normal distribution of residuals

### Evaluation Metrics
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Root Mean Square Error (RMSE)**: Square root of MSE
- **Adjusted R²**: R² adjusted for number of predictors

### Advantages & Disadvantages
**Pros:**
- Simple and interpretable
- Fast training and prediction
- No hyperparameter tuning needed
- Works well with linear relationships

**Cons:**
- Assumes linear relationship
- Sensitive to outliers
- Requires feature scaling
- Poor performance with non-linear data

---

## 2. Logistic Regression

### Interactive Sigmoid Function Visualization

<div style="width: 100%; height: 400px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="logisticCanvas" width="800" height="400"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Logistic Regression - Sigmoid Function</strong><br>
<label>Weight (β₁): <input type="range" id="weightSlider" min="-5" max="5" step="0.1" value="1"></label> <span id="weightValue">1.0</span><br>
<label>Bias (β₀): <input type="range" id="biasSlider" min="-5" max="5" step="0.1" value="0"></label> <span id="biasValue">0.0</span><br>
<span id="sigmoidEquation">σ(z) = 1 / (1 + e^(-(1.0x + 0.0)))</span>
</div>
</div>

<script>
const logisticCanvas = document.getElementById('logisticCanvas');
const logisticCtx = logisticCanvas.getContext('2d');
const weightSlider = document.getElementById('weightSlider');
const biasSlider = document.getElementById('biasSlider');

function sigmoid(x, weight, bias) {
    const z = weight * x + bias;
    return 1 / (1 + Math.exp(-z));
}

function drawLogistic() {
    const weight = parseFloat(weightSlider.value);
    const bias = parseFloat(biasSlider.value);
    
    logisticCtx.clearRect(0, 0, logisticCanvas.width, logisticCanvas.height);
    
    // Draw grid
    logisticCtx.strokeStyle = '#f0f0f0';
    logisticCtx.lineWidth = 1;
    
    // Vertical lines
    for(let i = 0; i <= logisticCanvas.width; i += 50) {
        logisticCtx.beginPath();
        logisticCtx.moveTo(i, 0);
        logisticCtx.lineTo(i, logisticCanvas.height);
        logisticCtx.stroke();
    }
    
    // Horizontal lines
    for(let i = 0; i <= logisticCanvas.height; i += 50) {
        logisticCtx.beginPath();
        logisticCtx.moveTo(0, i);
        logisticCtx.lineTo(logisticCanvas.width, i);
        logisticCtx.stroke();
    }
    
    // Draw axes
    logisticCtx.strokeStyle = '#333';
    logisticCtx.lineWidth = 2;
    
    // Y-axis (center)
    logisticCtx.beginPath();
    logisticCtx.moveTo(logisticCanvas.width/2, 0);
    logisticCtx.lineTo(logisticCanvas.width/2, logisticCanvas.height);
    logisticCtx.stroke();
    
    // X-axis 
    logisticCtx.beginPath();
    logisticCtx.moveTo(0, logisticCanvas.height/2);
    logisticCtx.lineTo(logisticCanvas.width, logisticCanvas.height/2);
    logisticCtx.stroke();
    
    // Draw sigmoid curve
    logisticCtx.strokeStyle = '#4CAF50';
    logisticCtx.lineWidth = 3;
    logisticCtx.beginPath();
    
    for(let x = 0; x < logisticCanvas.width; x++) {
        // Convert canvas x to actual x value (-10 to 10)
        const actualX = (x - logisticCanvas.width/2) / 40;
        const y = sigmoid(actualX, weight, bias);
        // Convert probability (0-1) to canvas y (flip because canvas y increases downward)
        const canvasY = logisticCanvas.height - (y * logisticCanvas.height);
        
        if(x === 0) {
            logisticCtx.moveTo(x, canvasY);
        } else {
            logisticCtx.lineTo(x, canvasY);
        }
    }
    logisticCtx.stroke();
    
    // Draw decision boundary at 0.5
    logisticCtx.strokeStyle = '#FF5722';
    logisticCtx.setLineDash([5, 5]);
    logisticCtx.beginPath();
    logisticCtx.moveTo(0, logisticCanvas.height/2);
    logisticCtx.lineTo(logisticCanvas.width, logisticCanvas.height/2);
    logisticCtx.stroke();
    logisticCtx.setLineDash([]);
    
    // Update labels
    document.getElementById('weightValue').textContent = weight.toFixed(1);
    document.getElementById('biasValue').textContent = bias.toFixed(1);
    document.getElementById('sigmoidEquation').textContent = 
        `σ(z) = 1 / (1 + e^(-(${weight.toFixed(1)}x + ${bias.toFixed(1)})))`;
}

weightSlider.addEventListener('input', drawLogistic);
biasSlider.addEventListener('input', drawLogistic);
drawLogistic();
</script>

### Mathematical Foundation
Logistic regression uses the sigmoid function to map any real number to a probability between 0 and 1:

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Linear Combination:**
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Probability:**
```
P(y=1|x) = σ(z) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

### Cost Function
Uses Maximum Likelihood Estimation with log-likelihood:
```
Cost = -[y log(p) + (1-y) log(1-p)]
```

### How It Works
1. **Calculate linear combination** z = β₀ + β₁x₁ + ... + βₙxₙ
2. **Apply sigmoid function** to get probabilities
3. **Make predictions** (threshold typically 0.5)
4. **Calculate cost** using log-likelihood
5. **Update weights** using gradient descent
6. **Repeat** until convergence

### Types
- **Binary Classification**: Two classes (0 or 1)
- **Multinomial**: Multiple classes (one-vs-rest or softmax)
- **Ordinal**: Ordered categories

### Evaluation Metrics
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

### Advantages & Disadvantages
**Pros:**
- Probabilistic output
- No assumptions about distribution
- Less prone to overfitting
- Interpretable coefficients

**Cons:**
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- Requires large sample sizes
- Can struggle with complex relationships

---

## 3. Decision Tree

### Interactive Decision Tree Visualization

<div style="width: 100%; height: 500px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="decisionTreeCanvas" width="800" height="500"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Decision Tree Classification</strong><br>
Click to add points: <br>
<button id="classA" style="background: #2196F3; color: white; border: none; padding: 5px;">Class A (Blue)</button>
<button id="classB" style="background: #f44336; color: white; border: none; padding: 5px;">Class B (Red)</button><br>
<button id="buildTree" style="margin-top: 5px; padding: 5px;">Build Tree</button>
<button id="clearTree" style="margin-top: 5px; padding: 5px;">Clear</button><br>
<span id="treeInfo">Points: 0</span>
</div>
</div>

<script>
const treeCanvas = document.getElementById('decisionTreeCanvas');
const treeCtx = treeCanvas.getContext('2d');
let treePoints = [];
let currentClass = 'A';
let treeSplits = [];

document.getElementById('classA').addEventListener('click', () => {
    currentClass = 'A';
    document.getElementById('classA').style.fontWeight = 'bold';
    document.getElementById('classB').style.fontWeight = 'normal';
});

document.getElementById('classB').addEventListener('click', () => {
    currentClass = 'B';
    document.getElementById('classB').style.fontWeight = 'bold';
    document.getElementById('classA').style.fontWeight = 'normal';
});

function calculateEntropy(points) {
    if(points.length === 0) return 0;
    const classA = points.filter(p => p.class === 'A').length;
    const classB = points.filter(p => p.class === 'B').length;
    const total = points.length;
    
    if(classA === 0 || classB === 0) return 0;
    
    const pA = classA / total;
    const pB = classB / total;
    
    return -(pA * Math.log2(pA) + pB * Math.log2(pB));
}

function findBestSplit(points) {
    let bestGain = 0;
    let bestSplit = null;
    const totalEntropy = calculateEntropy(points);
    
    // Try vertical splits (x-axis)
    for(let i = 50; i < treeCanvas.width - 50; i += 20) {
        const left = points.filter(p => p.x < i);
        const right = points.filter(p => p.x >= i);
        
        if(left.length === 0 || right.length === 0) continue;
        
        const weightedEntropy = (left.length / points.length) * calculateEntropy(left) +
                               (right.length / points.length) * calculateEntropy(right);
        
        const gain = totalEntropy - weightedEntropy;
        
        if(gain > bestGain) {
            bestGain = gain;
            bestSplit = {type: 'vertical', position: i, gain: gain};
        }
    }
    
    // Try horizontal splits (y-axis)
    for(let i = 50; i < treeCanvas.height - 50; i += 20) {
        const top = points.filter(p => p.y < i);
        const bottom = points.filter(p => p.y >= i);
        
        if(top.length === 0 || bottom.length === 0) continue;
        
        const weightedEntropy = (top.length / points.length) * calculateEntropy(top) +
                               (bottom.length / points.length) * calculateEntropy(bottom);
        
        const gain = totalEntropy - weightedEntropy;
        
        if(gain > bestGain) {
            bestGain = gain;
            bestSplit = {type: 'horizontal', position: i, gain: gain};
        }
    }
    
    return bestSplit;
}

function buildDecisionTree() {
    treeSplits = [];
    let regions = [{points: [...treePoints], bounds: {x1: 0, y1: 0, x2: treeCanvas.width, y2: treeCanvas.height}}];
    
    for(let depth = 0; depth < 3; depth++) {
        let newRegions = [];
        
        for(let region of regions) {
            const split = findBestSplit(region.points);
            
            if(split && split.gain > 0.1) {
                treeSplits.push({
                    type: split.type,
                    position: split.position,
                    bounds: region.bounds
                });
                
                if(split.type === 'vertical') {
                    const left = region.points.filter(p => p.x < split.position);
                    const right = region.points.filter(p => p.x >= split.position);
                    
                    newRegions.push({
                        points: left,
                        bounds: {x1: region.bounds.x1, y1: region.bounds.y1, x2: split.position, y2: region.bounds.y2}
                    });
                    newRegions.push({
                        points: right,
                        bounds: {x1: split.position, y1: region.bounds.y1, x2: region.bounds.x2, y2: region.bounds.y2}
                    });
                } else {
                    const top = region.points.filter(p => p.y < split.position);
                    const bottom = region.points.filter(p => p.y >= split.position);
                    
                    newRegions.push({
                        points: top,
                        bounds: {x1: region.bounds.x1, y1: region.bounds.y1, x2: region.bounds.x2, y2: split.position}
                    });
                    newRegions.push({
                        points: bottom,
                        bounds: {x1: region.bounds.x1, y1: split.position, x2: region.bounds.x2, y2: region.bounds.y2}
                    });
                }
            } else {
                newRegions.push(region);
            }
        }
        
        regions = newRegions;
    }
    
    drawDecisionTree();
}

function drawDecisionTree() {
    treeCtx.clearRect(0, 0, treeCanvas.width, treeCanvas.height);
    
    // Draw grid
    treeCtx.strokeStyle = '#f0f0f0';
    treeCtx.lineWidth = 1;
    for(let i = 0; i <= treeCanvas.width; i += 50) {
        treeCtx.beginPath();
        treeCtx.moveTo(i, 0);
        treeCtx.lineTo(i, treeCanvas.height);
        treeCtx.stroke();
    }
    for(let i = 0; i <= treeCanvas.height; i += 50) {
        treeCtx.beginPath();
        treeCtx.moveTo(0, i);
        treeCtx.lineTo(treeCanvas.width, i);
        treeCtx.stroke();
    }
    
    // Draw decision boundaries
    treeCtx.strokeStyle = '#4CAF50';
    treeCtx.lineWidth = 3;
    
    treeSplits.forEach(split => {
        treeCtx.beginPath();
        if(split.type === 'vertical') {
            treeCtx.moveTo(split.position, split.bounds.y1);
            treeCtx.lineTo(split.position, split.bounds.y2);
        } else {
            treeCtx.moveTo(split.bounds.x1, split.position);
            treeCtx.lineTo(split.bounds.x2, split.position);
        }
        treeCtx.stroke();
    });
    
    // Draw points
    treePoints.forEach(point => {
        treeCtx.fillStyle = point.class === 'A' ? '#2196F3' : '#f44336';
        treeCtx.beginPath();
        treeCtx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
        treeCtx.fill();
    });
    
    document.getElementById('treeInfo').textContent = 
        `Points: ${treePoints.length}, Splits: ${treeSplits.length}`;
}

treeCanvas.addEventListener('click', (e) => {
    const rect = treeCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    treePoints.push({x, y, class: currentClass});
    drawDecisionTree();
});

document.getElementById('buildTree').addEventListener('click', buildDecisionTree);
document.getElementById('clearTree').addEventListener('click', () => {
    treePoints = [];
    treeSplits = [];
    drawDecisionTree();
});

// Initialize with some sample points
treePoints = [
    {x: 200, y: 150, class: 'A'}, {x: 250, y: 180, class: 'A'},
    {x: 180, y: 200, class: 'A'}, {x: 500, y: 300, class: 'B'},
    {x: 550, y: 280, class: 'B'}, {x: 480, y: 320, class: 'B'}
];

drawDecisionTree();
</script>

### Mathematical Foundation
Decision trees use information theory concepts:

**Entropy (measure of impurity):**
```
H(S) = -Σᵢ₌₁ᶜ pᵢ log₂(pᵢ)
```

**Information Gain:**
```
IG(S,A) = H(S) - Σᵥ∈Values(A) (|Sᵥ|/|S|) × H(Sᵥ)
```

**Gini Impurity:**
```
Gini(S) = 1 - Σᵢ₌₁ᶜ pᵢ²
```

Where:
- S = dataset
- c = number of classes
- pᵢ = proportion of class i
- A = attribute/feature

### How It Works
1. **Start with root node** containing all training data
2. **Calculate impurity** for current node
3. **For each feature**, calculate information gain or Gini gain
4. **Select best feature** that maximizes information gain
5. **Split data** based on selected feature
6. **Create child nodes** and repeat recursively
7. **Stop when** stopping criteria met (max depth, min samples, etc.)

### Tree Construction Algorithm (ID3/C4.5/CART)
```
function BuildTree(dataset, features):
    if all examples have same class:
        return leaf node with that class
    if no features remaining:
        return leaf node with majority class
    
    best_feature = select_best_feature(dataset, features)
    tree = create_node(best_feature)
    
    for each value v of best_feature:
        subset = examples with best_feature = v
        subtree = BuildTree(subset, features - best_feature)
        add subtree to tree
    
    return tree
```

### Pruning Techniques
- **Pre-pruning**: Stop early (max depth, min samples)
- **Post-pruning**: Build full tree, then remove branches
- **Cost Complexity Pruning**: Balance tree complexity and accuracy

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision, Recall, F1**: Class-specific performance
- **Feature Importance**: How much each feature contributes
- **Tree Depth**: Complexity measure
- **Number of Leaves**: Model complexity

### Advantages & Disadvantages
**Pros:**
- Easy to understand and visualize
- Requires little data preparation
- Handles both numerical and categorical data
- Can capture non-linear relationships
- Provides feature importance

**Cons:**
- Prone to overfitting
- Unstable (small data changes = different tree)
- Biased toward features with more levels
- Can create overly complex trees

---

## 4. Support Vector Machine (SVM)

### Interactive SVM Visualization

<div style="width: 100%; height: 500px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="svmCanvas" width="800" height="500"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Support Vector Machine</strong><br>
Click to add points:<br>
<button id="svmClassA" style="background: #2196F3; color: white; border: none; padding: 5px;">Class +1 (Blue)</button>
<button id="svmClassB" style="background: #f44336; color: white; border: none; padding: 5px;">Class -1 (Red)</button><br>
<label>C Parameter: <input type="range" id="cParameter" min="0.1" max="10" step="0.1" value="1"></label> <span id="cValue">1.0</span><br>
<button id="trainSVM" style="margin-top: 5px; padding: 5px;">Train SVM</button>
<button id="clearSVM" style="margin-top: 5px; padding: 5px;">Clear</button><br>
<span id="svmInfo">Margin: 0</span>
</div>
</div>

<script>
const svmCanvas = document.getElementById('svmCanvas');
const svmCtx = svmCanvas.getContext('2d');
let svmPoints = [];
let currentSvmClass = 1;
let svmModel = null;

document.getElementById('svmClassA').addEventListener('click', () => {
    currentSvmClass = 1;
    document.getElementById('svmClassA').style.fontWeight = 'bold';
    document.getElementById('svmClassB').style.fontWeight = 'normal';
});

document.getElementById('svmClassB').addEventListener('click', () => {
    currentSvmClass = -1;
    document.getElementById('svmClassB').style.fontWeight = 'bold';
    document.getElementById('svmClassA').style.fontWeight = 'normal';
});

function trainSVM() {
    if(svmPoints.length < 2) return;
    
    const C = parseFloat(document.getElementById('cParameter').value);
    
    // Simple linear SVM implementation (simplified)
    // Find the hyperplane that separates the classes
    let bestW = null;
    let bestB = 0;
    let maxMargin = 0;
    
    // Try different hyperplane orientations
    for(let angle = 0; angle < Math.PI; angle += 0.1) {
        const w = {x: Math.cos(angle), y: Math.sin(angle)};
        
        // Project all points onto the normal vector
        const projections = svmPoints.map(p => ({
            projection: p.x * w.x + p.y * w.y,
            class: p.class
        }));
        
        // Find the separating hyperplane
        const pos = projections.filter(p => p.class === 1).map(p => p.projection);
        const neg = projections.filter(p => p.class === -1).map(p => p.projection);
        
        if(pos.length === 0 || neg.length === 0) continue;
        
        const minPos = Math.min(...pos);
        const maxNeg = Math.max(...neg);
        
        if(minPos > maxNeg) {
            const margin = (minPos - maxNeg) / 2;
            if(margin > maxMargin) {
                maxMargin = margin;
                bestW = w;
                bestB = -(minPos + maxNeg) / 2;
            }
        }
    }
    
    svmModel = {w: bestW, b: bestB, margin: maxMargin};
    drawSVM();
}

function drawSVM() {
    svmCtx.clearRect(0, 0, svmCanvas.width, svmCanvas.height);
    
    // Draw grid
    svmCtx.strokeStyle = '#f0f0f0';
    svmCtx.lineWidth = 1;
    for(let i = 0; i <= svmCanvas.width; i += 50) {
        svmCtx.beginPath();
        svmCtx.moveTo(i, 0);
        svmCtx.lineTo(i, svmCanvas.height);
        svmCtx.stroke();
    }
    for(let i = 0; i <= svmCanvas.height; i += 50) {
        svmCtx.beginPath();
        svmCtx.moveTo(0, i);
        svmCtx.lineTo(svmCanvas.width, i);
        svmCtx.stroke();
    }
    
    // Draw SVM hyperplane and margins
    if(svmModel && svmModel.w) {
        const w = svmModel.w;
        const b = svmModel.b;
        
        // Calculate line endpoints
        // w.x * x + w.y * y + b = 0
        // y = -(w.x * x + b) / w.y
        
        if(Math.abs(w.y) > 0.001) {
            const y1 = -(w.x * 0 + b) / w.y;
            const y2 = -(w.x * svmCanvas.width + b) / w.y;
            
            // Decision boundary
            svmCtx.strokeStyle = '#4CAF50';
            svmCtx.lineWidth = 3;
            svmCtx.beginPath();
            svmCtx.moveTo(0, y1);
            svmCtx.lineTo(svmCanvas.width, y2);
            svmCtx.stroke();
            
            // Margin boundaries
            svmCtx.strokeStyle = '#4CAF50';
            svmCtx.lineWidth = 1;
            svmCtx.setLineDash([5, 5]);
            
            const margin = svmModel.margin;
            const y1_pos = -(w.x * 0 + b - margin) / w.y;
            const y2_pos = -(w.x * svmCanvas.width + b - margin) / w.y;
            const y1_neg = -(w.x * 0 + b + margin) / w.y;
            const y2_neg = -(w.x * svmCanvas.width + b + margin) / w.y;
            
            svmCtx.beginPath();
            svmCtx.moveTo(0, y1_pos);
            svmCtx.lineTo(svmCanvas.width, y2_pos);
            svmCtx.stroke();
            
            svmCtx.beginPath();
            svmCtx.moveTo(0, y1_neg);
            svmCtx.lineTo(svmCanvas.width, y2_neg);
            svmCtx.stroke();
            
            svmCtx.setLineDash([]);
        }
        
        document.getElementById('svmInfo').textContent = 
            `Margin: ${svmModel.margin.toFixed(2)}`;
    }
    
    // Draw points and highlight support vectors
    svmPoints.forEach(point => {
        // Check if point is a support vector (close to margin)
        let isSupport = false;
        if(svmModel && svmModel.w) {
            const distance = Math.abs(svmModel.w.x * point.x + svmModel.w.y * point.y + svmModel.b);
            isSupport = distance < svmModel.margin + 10; // Some tolerance
        }
        
        svmCtx.fillStyle = point.class === 1 ? '#2196F3' : '#f44336';
        svmCtx.beginPath();
        svmCtx.arc(point.x, point.y, isSupport ? 10 : 6, 0, 2 * Math.PI);
        svmCtx.fill();
        
        // Draw support vector outline
        if(isSupport) {
            svmCtx.strokeStyle = '#000';
            svmCtx.lineWidth = 2;
            svmCtx.stroke();
        }
    });
    
    document.getElementById('cValue').textContent = 
        document.getElementById('cParameter').value;
}

svmCanvas.addEventListener('click', (e) => {
    const rect = svmCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    svmPoints.push({x, y, class: currentSvmClass});
    drawSVM();
});

document.getElementById('trainSVM').addEventListener('click', trainSVM);
document.getElementById('clearSVM').addEventListener('click', () => {
    svmPoints = [];
    svmModel = null;
    drawSVM();
});

document.getElementById('cParameter').addEventListener('input', () => {
    if(svmModel) trainSVM();
});

// Initialize with sample points
svmPoints = [
    {x: 200, y: 200, class: 1}, {x: 180, y: 220, class: 1}, {x: 220, y: 180, class: 1},
    {x: 400, y: 300, class: -1}, {x: 420, y: 320, class: -1}, {x: 380, y: 280, class: -1}
];

drawSVM();
</script>

### Mathematical Foundation
SVM finds the optimal hyperplane that separates classes with maximum margin.

**Linear SVM Objective:**
```
Minimize: (1/2)||w||² + C Σᵢ₌₁ⁿ ξᵢ
```

**Subject to:**
```
yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Decision Function:**
```
f(x) = sign(w·x + b) = sign(Σᵢ₌₁ⁿ αᵢyᵢK(xᵢ,x) + b)
```

Where:
- w = weight vector
- b = bias term
- C = regularization parameter
- ξᵢ = slack variables
- αᵢ = Lagrange multipliers
- K(xᵢ,x) = kernel function

### Kernel Functions
**Linear Kernel:**
```
K(xᵢ, xⱼ) = xᵢ · xⱼ
```

**Polynomial Kernel:**
```
K(xᵢ, xⱼ) = (γxᵢ · xⱼ + r)^d
```

**RBF (Gaussian) Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```

**Sigmoid Kernel:**
```
K(xᵢ, xⱼ) = tanh(γxᵢ · xⱼ + r)
```

### How It Works
1. **Map data** to higher dimensional space (if using kernel)
2. **Find support vectors** (data points closest to decision boundary)
3. **Solve quadratic optimization** problem to find optimal hyperplane
4. **Construct decision function** using support vectors
5. **Classify new points** based on which side of hyperplane they fall

### Key Concepts
- **Margin**: Distance between hyperplane and nearest points
- **Support Vectors**: Training points that define the margin
- **Kernel Trick**: Implicit mapping to higher dimensions
- **Soft Margin**: Allows some misclassification (C parameter)

### Hyperparameters
- **C**: Regularization (low = wider margin, high = stricter)
- **γ (gamma)**: Kernel coefficient (low = far influence, high = close influence)
- **kernel**: Type of kernel function
- **degree**: Degree for polynomial kernel

### Evaluation Metrics
- **Accuracy**: Overall performance
- **Precision, Recall, F1**: Per-class metrics
- **Support Vector Count**: Model complexity
- **Margin Width**: Generalization indicator

### Advantages & Disadvantages
**Pros:**
- Effective in high dimensions
- Memory efficient (uses support vectors)
- Versatile (different kernels)
- Works well with small datasets

**Cons:**
- Slow on large datasets
- Sensitive to feature scaling
- No probabilistic output
- Difficult to interpret
- Choice of kernel and parameters crucial

---

## 5. Random Forest

### Mathematical Foundation
Random Forest combines multiple decision trees using bagging and random feature selection:

**Prediction (Classification):**
```
ŷ = mode{T₁(x), T₂(x), ..., Tₙ(x)}
```

**Prediction (Regression):**
```
ŷ = (1/n) Σᵢ₌₁ⁿ Tᵢ(x)
```

**Out-of-Bag Error:**
```
OOB Error = (1/n) Σᵢ₌₁ⁿ I(yᵢ ≠ ŷᵢ^(OOB))
```

Where:
- Tᵢ(x) = prediction from tree i
- n = number of trees
- ŷᵢ^(OOB) = prediction using only trees where xᵢ was out-of-bag

### How It Works
1. **Bootstrap Sampling**: Create n bootstrap samples from training data
2. **Random Feature Selection**: At each split, select random subset of features
3. **Build Trees**: Train decision tree on each bootstrap sample
4. **Combine Predictions**: 
   - Classification: Majority voting
   - Regression: Average predictions
5. **Calculate OOB Error**: Use out-of-bag samples for error estimation

### Algorithm Steps
```
function RandomForest(dataset, n_trees, n_features):
    forest = []
    
    for i in range(n_trees):
        # Bootstrap sampling
        bootstrap_sample = sample_with_replacement(dataset)
        
        # Build tree with random feature selection
        tree = DecisionTree(bootstrap_sample, n_features)
        forest.append(tree)
    
    return forest

function Predict(forest, x):
    predictions = []
    for tree in forest:
        predictions.append(tree.predict(x))
    
    return majority_vote(predictions)  # or average for regression
```

### Key Parameters
- **n_estimators**: Number of trees
- **max_features**: Number of features for each split
- **max_depth**: Maximum tree depth
- **min_samples_split**: Minimum samples to split
- **min_samples_leaf**: Minimum samples in leaf
- **bootstrap**: Whether to use bootstrap sampling

### Feature Importance
Calculated based on how much each feature decreases impurity:
```
Importance(feature) = Σ(trees) (decrease in impurity) / n_trees
```

### Evaluation Metrics
- **Accuracy/RMSE**: Overall performance
- **OOB Score**: Out-of-bag accuracy/error
- **Feature Importance**: Ranking of feature contributions
- **Precision, Recall, F1**: Classification metrics

### Advantages & Disadvantages
**Pros:**
- Reduces overfitting compared to single trees
- Handles missing values
- Provides feature importance
- Works with both classification and regression
- Robust to outliers
- Requires minimal hyperparameter tuning

**Cons:**
- Less interpretable than single tree
- Can overfit with very noisy data
- Biased toward categorical variables with more categories
- Memory intensive
- Not optimal for linear relationships

---

## 6. Gradient Boosting

### Gradient Boosting Sequential Learning

<div style="width: 100%; height: 500px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="gradientBoostingCanvas" width="800" height="500"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; max-width: 200px;">
<strong>Gradient Boosting</strong><br>
<label>Learning Rate: <input type="range" id="learningRate" min="0.1" max="1" step="0.1" value="0.3"></label> <span id="lrValue">0.3</span><br>
<label>Iterations: <input type="range" id="iterations" min="1" max="10" value="5"></label> <span id="iterValue">5</span><br>
<button id="stepBoost" style="margin-top: 5px; padding: 5px;">Step Through</button>
<button id="trainBoost" style="margin-top: 5px; padding: 5px;">Train All</button>
<button id="resetBoost" style="margin-top: 5px; padding: 5px;">Reset</button><br>
<span id="boostInfo">Step: 0, Error: 0</span>
</div>
</div>

<script>
const gbCanvas = document.getElementById('gradientBoostingCanvas');
const gbCtx = gbCanvas.getContext('2d');

// Generate regression data
let gbData = [];
for(let i = 0; i < 50; i++) {
    const x = (i / 49) * gbCanvas.width;
    // True function: sine wave with noise
    const y = gbCanvas.height/2 + Math.sin(x * 0.02) * 100 + (Math.random() - 0.5) * 40;
    gbData.push({x: x, y: y});
}

let boosters = [];
let currentStep = 0;
let residuals = [];

function initializeGradientBoosting() {
    boosters = [];
    currentStep = 0;
    
    // Initialize with mean prediction
    const meanY = gbData.reduce((sum, d) => sum + d.y, 0) / gbData.length;
    boosters.push({
        type: 'constant',
        value: meanY,
        predictions: gbData.map(() => meanY)
    });
    
    calculateResiduals();
    drawGradientBoosting();
}

function calculateResiduals() {
    // Calculate current ensemble predictions
    const ensemblePredictions = gbData.map((_, i) => {
        return boosters.reduce((sum, booster) => {
            if(booster.type === 'constant') {
                return sum + booster.value;
            } else {
                return sum + booster.predictions[i];
            }
        }, 0);
    });
    
    // Calculate residuals
    residuals = gbData.map((d, i) => ({
        x: d.x,
        y: d.y - ensemblePredictions[i],
        actual: d.y,
        predicted: ensemblePredictions[i]
    }));
}

function fitWeakLearner() {
    // Simple linear regression on residuals
    const n = residuals.length;
    const sumX = residuals.reduce((sum, r) => sum + r.x, 0);
    const sumY = residuals.reduce((sum, r) => sum + r.y, 0);
    const sumXY = residuals.reduce((sum, r) => sum + r.x * r.y, 0);
    const sumX2 = residuals.reduce((sum, r) => sum + r.x * r.x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    const predictions = residuals.map(r => slope * r.x + intercept);
    
    return {
        type: 'linear',
        slope: slope,
        intercept: intercept,
        predictions: predictions
    };
}

function stepGradientBoosting() {
    const maxIterations = parseInt(document.getElementById('iterations').value);
    if(currentStep >= maxIterations) return;
    
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    
    // Fit weak learner to residuals
    const weakLearner = fitWeakLearner();
    
    // Scale by learning rate
    weakLearner.predictions = weakLearner.predictions.map(p => p * learningRate);
    weakLearner.slope *= learningRate;
    weakLearner.intercept *= learningRate;
    
    boosters.push(weakLearner);
    currentStep++;
    
    calculateResiduals();
    drawGradientBoosting();
}

function trainAllGradientBoosting() {
    const maxIterations = parseInt(document.getElementById('iterations').value);
    while(currentStep < maxIterations) {
        stepGradientBoosting();
    }
}

function drawGradientBoosting() {
    gbCtx.clearRect(0, 0, gbCanvas.width, gbCanvas.height);
    
    // Draw grid
    gbCtx.strokeStyle = '#f0f0f0';
    gbCtx.lineWidth = 1;
    for(let i = 0; i <= gbCanvas.width; i += 50) {
        gbCtx.beginPath();
        gbCtx.moveTo(i, 0);
        gbCtx.lineTo(i, gbCanvas.height);
        gbCtx.stroke();
    }
    for(let i = 0; i <= gbCanvas.height; i += 50) {
        gbCtx.beginPath();
        gbCtx.moveTo(0, i);
        gbCtx.lineTo(gbCanvas.width, i);
        gbCtx.stroke();
    }
    
    // Draw original data points
    gbCtx.fillStyle = '#2196F3';
    gbData.forEach(d => {
        gbCtx.beginPath();
        gbCtx.arc(d.x, d.y, 4, 0, 2 * Math.PI);
        gbCtx.fill();
    });
    
    // Draw ensemble prediction
    if(boosters.length > 0) {
        gbCtx.strokeStyle = '#f44336';
        gbCtx.lineWidth = 3;
        gbCtx.beginPath();
        
        for(let x = 0; x < gbCanvas.width; x += 2) {
            let prediction = 0;
            
            boosters.forEach(booster => {
                if(booster.type === 'constant') {
                    prediction += booster.value;
                } else {
                    prediction += booster.slope * x + booster.intercept;
                }
            });
            
            if(x === 0) {
                gbCtx.moveTo(x, prediction);
            } else {
                gbCtx.lineTo(x, prediction);
            }
        }
        gbCtx.stroke();
        
        // Draw current weak learner if exists
        if(boosters.length > 1) {
            const currentLearner = boosters[boosters.length - 1];
            if(currentLearner.type === 'linear') {
                gbCtx.strokeStyle = '#4CAF50';
                gbCtx.lineWidth = 2;
                gbCtx.setLineDash([5, 5]);
                gbCtx.beginPath();
                gbCtx.moveTo(0, currentLearner.intercept);
                gbCtx.lineTo(gbCanvas.width, currentLearner.slope * gbCanvas.width + currentLearner.intercept);
                gbCtx.stroke();
                gbCtx.setLineDash([]);
            }
        }
        
        // Calculate and display error
        const mse = residuals.reduce((sum, r) => sum + r.y * r.y, 0) / residuals.length;
        document.getElementById('boostInfo').textContent = 
            `Step: ${currentStep}, MSE: ${mse.toFixed(2)}`;
    }
    
    // Update slider displays
    document.getElementById('lrValue').textContent = document.getElementById('learningRate').value;
    document.getElementById('iterValue').textContent = document.getElementById('iterations').value;
}

document.getElementById('stepBoost').addEventListener('click', stepGradientBoosting);
document.getElementById('trainBoost').addEventListener('click', trainAllGradientBoosting);
document.getElementById('resetBoost').addEventListener('click', initializeGradientBoosting);

document.getElementById('learningRate').addEventListener('input', () => {
    if(boosters.length > 0) initializeGradientBoosting();
});

document.getElementById('iterations').addEventListener('input', drawGradientBoosting);

// Initialize
initializeGradientBoosting();
</script>

---

## Algorithm Comparison Visualization

<div style="width: 100%; height: 400px; border: 1px solid #ccc; margin: 20px 0; position: relative;">
<canvas id="comparisonCanvas" width="800" height="400"></canvas>
<div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<strong>Algorithm Performance Comparison</strong><br>
<select id="datasetType">
<option value="linear">Linear Data</option>
<option value="nonlinear">Non-linear Data</option>
<option value="noisy">Noisy Data</option>
<option value="classification">Classification</option>
</select>
<button id="generateData" style="margin-left: 10px;">Generate Dataset</button>
</div>
<div style="position: absolute; bottom: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px;">
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 12px;">
<div><span style="color: #2196F3;">●</span> Linear Regression</div>
<div><span style="color: #f44336;">●</span> Decision Tree</div>
<div><span style="color: #4CAF50;">●</span> Random Forest</div>
<div><span style="color: #FF9800;">●</span> SVM</div>
<div><span style="color: #9C27B0;">●</span> Gradient Boosting</div>
<div><span style="color: #607D8B;">●</span> Logistic Regression</div>
</div>
</div>
</div>

<script>
const compCanvas = document.getElementById('comparisonCanvas');
const compCtx = compCanvas.getContext('2d');
let compData = [];

function generateComparisonData() {
    const datasetType = document.getElementById('datasetType').value;
    compData = [];
    
    const numPoints = 100;
    
    for(let i = 0; i < numPoints; i++) {
        const x = (i / (numPoints - 1)) * compCanvas.width;
        let y, label;
        
        switch(datasetType) {
            case 'linear':
                y = 100 + 0.5 * x + (Math.random() - 0.5) * 50;
                label = y > compCanvas.height / 2 ? 1 : 0;
                break;
            case 'nonlinear':
                y = compCanvas.height/2 + Math.sin(x * 0.02) * 100 + (Math.random() - 0.5) * 30;
                label = y > compCanvas.height / 2 ? 1 : 0;
                break;
            case 'noisy':
                y = 200 + 0.3 * x + (Math.random() - 0.5) * 150;
                label = y > compCanvas.height / 2 ? 1 : 0;
                break;
            case 'classification':
                const centerX = compCanvas.width / 2;
                const centerY = compCanvas.height / 2;
                const distance = Math.sqrt((x - centerX) ** 2 + ((i % 20) * 20 - centerY) ** 2);
                y = (i % 20) * 20;
                label = distance < 150 ? 1 : 0;
                break;
        }
        
        compData.push({x: x, y: Math.max(0, Math.min(compCanvas.height, y)), label: label});
    }
    
    drawComparison();
}

function drawComparison() {
    compCtx.clearRect(0, 0, compCanvas.width, compCanvas.height);
    
    // Draw grid
    compCtx.strokeStyle = '#f0f0f0';
    compCtx.lineWidth = 1;
    for(let i = 0; i <= compCanvas.width; i += 50) {
        compCtx.beginPath();
        compCtx.moveTo(i, 0);
        compCtx.lineTo(i, compCanvas.height);
        compCtx.stroke();
    }
    for(let i = 0; i <= compCanvas.height; i += 50) {
        compCtx.beginPath();
        compCtx.moveTo(0, i);
        compCtx.lineTo(compCanvas.width, i);
        compCtx.stroke();
    }
    
    // Draw data points
    compData.forEach(d => {
        compCtx.fillStyle = d.label === 1 ? '#2196F3' : '#f44336';
        compCtx.beginPath();
        compCtx.arc(d.x, d.y, 3, 0, 2 * Math.PI);
        compCtx.fill();
    });
    
    // Simulate different algorithm predictions (simplified visualizations)
    const algorithms = [
        {name: 'Linear', color: '#2196F3', style: 'linear'},
        {name: 'Tree', color: '#f44336', style: 'steps'},
        {name: 'Forest', color: '#4CAF50', style: 'smooth'},
        {name: 'SVM', color: '#FF9800', style: 'linear'},
        {name: 'Boosting', color: '#9C27B0', style: 'smooth'}
    ];
    
    algorithms.forEach((alg, index) => {
        compCtx.strokeStyle = alg.color;
        compCtx.lineWidth = 2;
        compCtx.globalAlpha = 0.7;
        compCtx.beginPath();
        
        for(let x = 0; x < compCanvas.width; x += 5) {
            let y;
            
            if(alg.style === 'linear') {
                y = compCanvas.height/2 + (x - compCanvas.width/2) * 0.3 + index * 20;
            } else if(alg.style === 'steps') {
                y = compCanvas.height/2 + Math.floor(x / 100) * 40 - 60 + index * 15;
            } else { // smooth
                y = compCanvas.height/2 + Math.sin(x * 0.015 + index) * 60 + index * 10;
            }
            
            y = Math.max(50, Math.min(compCanvas.height - 50, y));
            
            if(x === 0) {
                compCtx.moveTo(x, y);
            } else {
                compCtx.lineTo(x, y);
            }
        }
        compCtx.stroke();
    });
    
    compCtx.globalAlpha = 1;
}

document.getElementById('generateData').addEventListener('click', generateComparisonData);
document.getElementById('datasetType').addEventListener('change', generateComparisonData);

// Initialize
generateComparisonData();
</script>

### Mathematical Foundation
Gradient boosting builds models sequentially, where each model corrects errors of previous models:

**Additive Model:**
```
F(x) = Σᵢ₌₁ᴹ γᵢhᵢ(x)
```

**Forward Stagewise Addition:**
```
Fₘ(x) = Fₘ₋₁(x) + γₘhₘ(x)
```

**Loss Function Minimization:**
```
γₘ, hₘ = argmin Σᵢ₌₁ⁿ L(yᵢ, Fₘ₋₁(xᵢ) + γh(xᵢ))
```

**Gradient Descent in Function Space:**
```
rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}
```

Where:
- F(x) = final ensemble model
- hᵢ(x) = individual weak learner
- γᵢ = learning rate for model i
- L = loss function
- rᵢₘ = residuals (negative gradients)

### Algorithm Steps
```
function GradientBoosting(dataset, n_estimators, learning_rate):
    # Initialize with constant prediction
    F₀(x) = argmin Σᵢ L(yᵢ, γ)
    
    for m in range(1, n_estimators + 1):
        # Calculate residuals (negative gradients)
        for i in range(n):
            rᵢₘ = -∂L(yᵢ, Fₘ₋₁(xᵢ))/∂Fₘ₋₁(xᵢ)
        
        # Fit weak learner to residuals
        hₘ = fit_weak_learner(X, residuals)
        
        # Find optimal step size
        γₘ = argmin Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + γhₘ(xᵢ))
        
        # Update model
        Fₘ(x) = Fₘ₋₁(x) + learning_rate × γₘ × hₘ(x)
    
    return Fₘ
```

### Common Loss Functions
**Regression:**
- Squared Error: L(y, F(x)) = (y - F(x))²/2
- Absolute Error: L(y, F(x)) = |y - F(x)|
- Huber Loss: Combination of squared and absolute error

**Classification:**
- Logistic Loss: L(y, F(x)) = log(1 + exp(-yF(x)))
- Exponential Loss: L(y, F(x)) = exp(-yF(x))

### Regularization Techniques
1. **Learning Rate (η)**: Controls contribution of each tree
2. **Tree Constraints**: Max depth, min samples per leaf
3. **Subsampling**: Use random subset of training data
4. **Feature Subsampling**: Random subset of features per tree
5. **Early Stopping**: Stop when validation error stops improving

### Popular Implementations
- **XGBoost**: Extreme Gradient Boosting
- **LightGBM**: Light Gradient Boosting Machine
- **CatBoost**: Categorical Boosting
- **Scikit-learn**: GradientBoostingClassifier/Regressor

### Key Hyperparameters
- **n_estimators**: Number of boosting stages
- **learning_rate**: Shrinks contribution of each tree
- **max_depth**: Maximum depth of individual trees
- **subsample**: Fraction of samples for each tree
- **min_samples_split**: Minimum samples to split node

### Evaluation Metrics
- **Training vs Validation Error**: Monitor overfitting
- **Feature Importance**: Based on splits and gain
- **Learning Curves**: Performance over iterations
- **Standard classification/regression metrics**

### Advantages & Disadvantages
**Pros:**
- High predictive accuracy
- Handles different data types well
- Provides feature importance
- Robust to outliers
- No need for data preprocessing

**Cons:**
- Prone to overfitting
- Computationally intensive
- Many hyperparameters to tune
- Sensitive to noisy data
- Sequential nature makes it hard to parallelize
- Less interpretable than simpler models

---

## Comparison Summary

| Algorithm | Type | Interpretability | Overfitting Risk | Performance | Training Speed |
|-----------|------|------------------|------------------|-------------|----------------|
| Linear Regression | Regression | High | Low | Good for linear | Fast |
| Logistic Regression | Classification | High | Low | Good for linear | Fast |
| Decision Tree | Both | High | High | Variable | Fast |
| SVM | Both | Low | Medium | Good | Slow |
| Random Forest | Both | Medium | Low | Good | Medium |
| Gradient Boosting | Both | Low | High | Excellent | Slow |

## When to Use Each Algorithm

- **Linear/Logistic Regression**: Simple baseline, interpretability needed, linear relationships
- **Decision Trees**: Need interpretability, mixed data types, non-linear relationships
- **SVM**: High-dimensional data, small datasets, need robust boundaries
- **Random Forest**: Good all-around performer, need feature importance, avoid overfitting
- **Gradient Boosting**: Maximum accuracy needed, have time for tuning, competition/production

## General Tips for Implementation

1. **Start Simple**: Begin with linear models, then increase complexity
2. **Cross-Validation**: Always use cross-validation for model selection
3. **Feature Engineering**: Often more important than algorithm choice
4. **Ensemble Methods**: Combine multiple algorithms for better performance
5. **Hyperparameter Tuning**: Use grid search or random search
6. **Monitor Overfitting**: Use validation curves and learning curves
7. **Scale Features**: Important for distance-based algorithms (SVM, logistic regression)
8. **Handle Missing Data**: Important preprocessing step
9. **Domain Knowledge**: Incorporate business understanding into model selection