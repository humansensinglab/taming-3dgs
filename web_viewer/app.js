const socket = new WebSocket('ws://localhost:8009');

// Get the canvas and setup WebGL context
const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl');

// Verify WebGL support
if (!gl) {
    alert('Unable to initialize WebGL. Your browser may not support it.');
    throw new Error('WebGL not supported');
}

// Vertex shader program
const vsSource = `
    attribute vec4 aVertexPosition;
    attribute vec2 aTextureCoord;
    varying highp vec2 vTextureCoord;

    void main(void) {
      gl_Position = aVertexPosition;
      vTextureCoord = aTextureCoord;
    }
`;

// Fragment shader program
const fsSource = `
    precision highp float;
    varying highp vec2 vTextureCoord;
    uniform sampler2D uSampler;

    void main(void) {
      gl_FragColor = texture2D(uSampler, vTextureCoord);
    }
`;

function initShaderProgram(gl, vsSource, fsSource) {
    // Function to load a shader, compile it, and check for errors
    function loadShader(gl, type, source) {
        const shader = gl.createShader(type);  // Create a new shader object
        gl.shaderSource(shader, source);       // Send the source to the shader object
        gl.compileShader(shader);              // Compile the shader program

        // Check if compilation was successful
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    // Load and compile the vertex and fragment shaders
    const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

    // Create the shader program
    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    // Check if creating the shader program failed
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
        return null;
    }

    return shaderProgram;
}

// Initialize a shader program; this is where all the lighting for the
// vertices and so forth is established.
const shaderProgram = initShaderProgram(gl, vsSource, fsSource);

// Collect all the info needed to use the shader program.
// Look up which attributes our shader program is using.
const programInfo = {
    program: shaderProgram,
    attribLocations: {
        vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
        textureCoord: gl.getAttribLocation(shaderProgram, 'aTextureCoord'),
    },
    uniformLocations: {
        uSampler: gl.getUniformLocation(shaderProgram, 'uSampler'),
    },
};

function initBuffers(gl) {
    // Create a buffer for the square's positions.
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Define the positions for a rectangle that covers the entire canvas.
    const positions = [
        -1.0,  1.0,
         1.0,  1.0,
        -1.0, -1.0,
         1.0, -1.0,
    ];

    // Pass the list of positions into WebGL to build the shape.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // Set up the texture coordinates buffer for the rectangle.
    const textureCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, textureCoordBuffer);

    const textureCoordinates = [
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ];

    // Pass the texture coordinates to WebGL.
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoordinates), gl.STATIC_DRAW);

    return {
        position: positionBuffer,
        textureCoord: textureCoordBuffer,
    };
}

function drawScene(gl, programInfo, buffers) {
    // Clear the canvas before drawing.
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Tell WebGL how to pull out the positions from the position buffer into the vertexPosition attribute.
    {
        const numComponents = 2;  // pull out 2 values per iteration
        const type = gl.FLOAT;    // the data in the buffer is 32bit floats
        const normalize = false;  // don't normalize
        const stride = 0;         // how many bytes to get from one set of values to the next
        const offset = 0;         // how many bytes inside the buffer to start from
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
        gl.vertexAttribPointer(
            programInfo.attribLocations.vertexPosition,
            numComponents,
            type,
            normalize,
            stride,
            offset);
        gl.enableVertexAttribArray(
            programInfo.attribLocations.vertexPosition);
    }

    // Tell WebGL how to pull out the texture coordinates from the texture coordinate buffer.
    {
        const numComponents = 2; // every coordinate consists of 2 values
        const type = gl.FLOAT;   // the data in the buffer is 32bit floats
        const normalize = false; // don't normalize
        const stride = 0;        // how many bytes to get from one set of values to the next
        const offset = 0;        // how many bytes inside the buffer to start from
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers.textureCoord);
        gl.vertexAttribPointer(
            programInfo.attribLocations.textureCoord,
            numComponents,
            type,
            normalize,
            stride,
            offset);
        gl.enableVertexAttribArray(
            programInfo.attribLocations.textureCoord);
    }

    // Specify the texture to use.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.useProgram(programInfo.program)
    gl.uniform1i(programInfo.uniformLocations.uSampler, 0);

    // Draw the rectangle.
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// Prepare and load texture
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);

// Set the parameters so we can render any size image.
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

const buffers = initBuffers(gl);

gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);


socket.binaryType = 'arraybuffer'; // Ensure you receive the data as ArrayBuffer

intToSend = 4;

function sendInteger(){
	intToSend = parseInt(document.getElementById('numberInput').value, 10);
    const arrayBuffer = new ArrayBuffer(4);
	const view = new DataView(arrayBuffer);
    view.setInt32(0, intToSend, false);
    socket.send(arrayBuffer);
};

socket.onopen = function (event) {
    console.log('Connected to WebSocket server.');

    // Optionally send binary data like a negative integer as previously discussed
	sendInteger();
};

socket.onmessage = function (event) {
    const buffer = event.data;

    // Use DataView to handle different data types in the same buffer
    const view = new DataView(buffer);

    // Read the two integers (assuming they are the first 8 bytes, 4 bytes each)
    const width = view.getInt32(0, true);  // Read first integer (little-endian)
    const height = view.getInt32(4, true);  // Read second integer (little-endian)
	
	const canvas = document.getElementById('glcanvas');
	if(canvas.width != width || canvas.height != height)
	{
		const dpr = window.devicePixelRatio || 1; // Get the device's pixel ratio

		// Set the drawing buffer's size
		canvas.width = width * dpr;
		canvas.height = height * dpr;
		
		gl.viewport(0, 0, width * dpr, height * dpr)
	}

    // Create a TypedArray for the remaining bytes (tensor data)
    // Adjust the offset and length according to the actual data size
    const data = new Uint8Array(buffer, 8);  // Skip the first 8 bytes
	
		 if (data.byteLength !== width * height * 3) {
	 console.log(data.byteLength, width * height * 3)
	 }

    //console.log("Data as array:", data);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, width, height, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
	
	drawScene(gl, programInfo, buffers);
	
	sendInteger();
};

socket.onerror = function (error) {
    console.error('WebSocket Error:', error);
};

socket.onclose = function (event) {
    console.log('WebSocket is closed now.');
};