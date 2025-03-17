const express = require('express');
const os = require('os');
const cors = require('cors');
const app = express();
const WebSocket = require('ws');
const dns = require('dns').promises;
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const axios = require('axios');
const path = require('path');
const { spawn } = require('child_process');

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const PORT = process.env.PORT || 3000;
const FLASK_API_URL = 'http://localhost:5000';

// Start the Flask API if not running
let pythonProcess = null;
function startPythonAPI() {
    const pythonScript = path.join(__dirname, 'threat_detection_api.py');
    console.log('Starting Python Flask API...');
    
    if (process.platform === 'win32') {
        pythonProcess = spawn('python', [pythonScript]);
    } else {
        pythonProcess = spawn('python3', [pythonScript]);
    }

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python API output: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python API error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python API process exited with code ${code}`);
        if (code !== 0) {
            console.log('Restarting Python API in 5 seconds...');
            setTimeout(startPythonAPI, 5000);
        }
    });
}

startPythonAPI();

process.on('exit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

// Change WebSocket to use the same port as the HTTP server
const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

const wss = new WebSocket.Server({ server });

async function getCPUUsageWindows() {
    try {
        // Using PowerShell command for more accurate CPU usage
        const command = `powershell "Get-Counter '\\Processor(_Total)\\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue"`;
        const { stdout } = await execPromise(command);
        return parseFloat(stdout);
    } catch (error) {
        console.error('Error getting CPU usage:', error);
        // Fallback to simpler method if PowerShell fails
        try {
            const { stdout } = await execPromise('wmic cpu get loadpercentage');
            const lines = stdout.trim().split('\n');
            const value = lines[1];
            return parseFloat(value);
        } catch (fallbackError) {
            console.error('Fallback CPU check failed:', fallbackError);
            return 0;
        }
    }
}

async function getCPUUsageUnix() {
    try {
        const { stdout } = await execPromise('top -bn1 | grep "Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk \'{print 100 - $1}\'');
        return parseFloat(stdout);
    } catch (error) {
        console.error('Error getting CPU usage:', error);
        return 0;
    }
}

async function getCPUUsage() {
    if (process.platform === 'win32') {
        return getCPUUsageWindows();
    } else {
        return getCPUUsageUnix();
    }
}

async function checkNetworkLatency() {
    try {
        const startTime = Date.now();
        await dns.resolve('google.com');
        const endTime = Date.now();
        return endTime - startTime;
    } catch (error) {
        console.error('Error checking network latency:', error);
        return 0;
    }
}

async function getSystemMetrics() {
    const cpuPercentage = await getCPUUsage();
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const memoryUsage = ((totalMem - freeMem) / totalMem) * 100;
    const networkLatency = await checkNetworkLatency();
    
    return {
        online: true,
        status: cpuPercentage > 80 ? 'Critical' : cpuPercentage > 60 ? 'Moderate' : 'Operational',
        cpuLoad: cpuPercentage.toFixed(2),
        storage: (100 - (freeMem / totalMem) * 100).toFixed(2),
        memory: memoryUsage.toFixed(2),
        networkLatency,
        networkInterfaces: os.networkInterfaces(),
        uptime: os.uptime(),
        platform: os.platform(),
        arch: os.arch(),
        totalMemory: (totalMem / (1024 * 1024 * 1024)).toFixed(2), // in GB
        freeMemory: (freeMem / (1024 * 1024 * 1024)).toFixed(2), // in GB
        cpuInfo: os.cpus(),
        hostname: os.hostname(),
        activeProcesses: process.pid,
        lastChecked: new Date()
    };
}

// WebSocket connection handler
wss.on('connection', (ws) => {
    console.log('Client connected');
    
    // Handle WebSocket errors
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        clearInterval(interval);
    });
    
    // Send system metrics every second
    const interval = setInterval(async () => {
        try {
            if (ws.readyState === WebSocket.OPEN) {
                const metrics = await getSystemMetrics();
                ws.send(JSON.stringify(metrics));
            } else if (ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
                console.log('Connection is closing or closed, clearing interval');
                clearInterval(interval);
            }
        } catch (error) {
            console.error('Error sending metrics via WebSocket:', error);
            // Don't clear the interval here, just log the error and continue trying
        }
    }, 1000);

    ws.on('close', () => {
        console.log('Client disconnected');
        clearInterval(interval);
    });
});

// REST endpoint for initial data
app.get('/api/system-metrics', async (req, res) => {
    const metrics = await getSystemMetrics();
    res.json(metrics);
});

// Proxy endpoint for threat detection
app.post('/api/detect-threat', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_API_URL}/api/detect`, req.body, {
            headers: { 'Content-Type': 'application/json' }
        });
        
        console.log('Threat detection result:', response.data);
        res.json(response.data);
    } catch (error) {
        console.error('Error detecting threat:', error.message);
        res.status(500).json({
            error: 'Failed to process image',
            details: error.message
        });
    }
});

// File upload endpoint for threat detection
app.post('/api/detect-threat/upload', async (req, res) => {
    try {
        // Set a longer timeout for the request to Flask API
        const response = await axios.post(`${FLASK_API_URL}/api/detect`, req.body, {
            headers: req.headers,
            timeout: 60000, // 60 second timeout for processing large images
            maxContentLength: 50 * 1024 * 1024, // 50MB max content length
            maxBodyLength: 50 * 1024 * 1024 // 50MB max body length
        });
        
        console.log('Threat detection result from upload:', response.data);
        res.json(response.data);
    } catch (error) {
        console.error('Error with file upload:', error.message);
        
        // Better error handling with specific error messages
        if (error.code === 'ECONNABORTED') {
            res.status(504).json({
                error: 'Request timed out while processing the image',
                details: 'The operation took too long to complete. Please try with a smaller image.'
            });
        } else if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            res.status(error.response.status).json({
                error: 'Error from detection service',
                details: error.response.data
            });
        } else {
            res.status(500).json({
                error: 'Failed to process uploaded file',
                details: error.message
            });
        }
    }
});