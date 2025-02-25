const express = require('express');
const os = require('os');
const cors = require('cors');
const app = express();
const WebSocket = require('ws');
const dns = require('dns').promises;
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;

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
    
    // Send system metrics every second
    const interval = setInterval(async () => {
        if (ws.readyState === WebSocket.OPEN) {
            const metrics = await getSystemMetrics();
            ws.send(JSON.stringify(metrics));
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