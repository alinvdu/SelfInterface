import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { AuthProvider } from './auth/AuthContext';
// import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AuthProvider>
		<App />
	</AuthProvider>
  </React.StrictMode>
);


// Just a little plugin that displays error on the iPad
	// let inlineConsole,
	// 	logCount = 0,
	// 	XMLHttpRequestCount = 0;
	// const consoleHeight = 240,
	// 	// watchEvents = ['click','focus','unfocus','blur','unblur','touchstart','touchend']
	// 	watchEvents = [],
	// 	startMs = new Date().getTime(),
	// 	oldLog = console.log,
	// 	oldDebug = console.debug,
	// 	oldWarn = console.warn,
	// 	oldInfo = console.info,
	// 	oldError = console.error,
	// 	consoleWrapper = document.createElement("div");

	// /** From: http://stackoverflow.com/questions/2234979/how-to-check-in-javascript-if-one-element-is-contained-within-another **/
	// const isDescendant = function (parent, child) {
	// 	let node = child.parentNode;
	// 	while (node !== null) {
	// 		if (node === parent) {
	// 			return true;
	// 		}
	// 		node = node.parentNode;
	// 	}
	// 	return false;
	// };

	// const clearConsole = function () {
	// 	inlineConsole.innerHTML = null;
	// };

	// const initInlineConsole = function () {
	// 	if (!inlineConsole) {
	// 		const consoleTitle = document.createElement("h3"),
	// 			clearButton = document.createElement("button");

	// 		inlineConsole = document.createElement("div");

	// 		consoleWrapper.appendChild(consoleTitle);
	// 		consoleWrapper.appendChild(inlineConsole);

	// 		consoleWrapper.style.backgroundColor = "#333333";
	// 		consoleWrapper.style.color = "#cccccc";
	// 		consoleWrapper.style.position = "fixed";
	// 		consoleWrapper.style.bottom = "0";
	// 		consoleWrapper.style.right = "0";
	// 		consoleWrapper.style.left = "0";
	// 		consoleWrapper.style.clear = "both";
	// 		consoleWrapper.classList.add("inline-console");

	// 		clearButton.innerHTML = "clear";
	// 		clearButton.style.fontSize = "0.5em";
	// 		clearButton.style.float = "right";
	// 		clearButton.style.color = "black";
	// 		clearButton.onclick = clearConsole;

	// 		consoleTitle.innerHTML = "Inline Console";
	// 		consoleTitle.style.padding = "0.3em";
	// 		consoleTitle.style.margin = "0";
	// 		consoleTitle.style.fontFamily = "monospace";
	// 		consoleTitle.appendChild(clearButton);

	// 		inlineConsole.style.backgroundColor = "black";
	// 		inlineConsole.style.border = "0";
	// 		inlineConsole.style.color = "#00ff00";
	// 		inlineConsole.style.height = consoleHeight + "px";
	// 		inlineConsole.style.resize = "none";
	// 		inlineConsole.style.overflowY = "auto";
	// 		inlineConsole.style.fontFamily = "monospace";
	// 	}
	// };

	// const toggleVisible = function (event, id, toggleElement) {
	// 	const element = document.getElementById(id);
	// 	const hiddenElement = document.getElementById(id + "-hidden");

	// 	if (element.style.display) {
	// 		element.style.display = null;
	// 		hiddenElement.style.display = "none";
	// 		toggleElement.innerHTML = "hide";
	// 	} else {
	// 		element.style.display = "none";
	// 		hiddenElement.style.display = null;
	// 		toggleElement.innerHTML = "show";
	// 	}
	// };

	// const sendMsg = function (type, args, color) {
	// 	const el = document.createElement("div"),
	// 		  eId = type + "-" + ++logCount,
	// 		  // Build a string from all arguments:
	// 		  allArgsString = args.map(arg => arg.toString()).join(", "),
	// 		  // Use a preview substring from all the arguments instead of just the first:
	// 		  preview = allArgsString.substring(0, 48),
	// 		  toggleElement = document.createElement("button"),
	// 		  hrElement = document.createElement("hr"),
	// 		  currentMs = new Date().getTime();
	// 	let nElement;
	
	// 	initInlineConsole();
	
	// 	el.style.clear = "both";
	// 	el.style.margin = "0";
	// 	el.style.padding = "0";
	// 	el.style.textAlign = "left";
	// 	if (color) {
	// 		el.style.color = color;
	// 	}
	
	// 	// Add Header using the combined arguments string
	// 	el.innerHTML +=
	// 		"<strong>" + type + "</strong> [" +
	// 		(currentMs - startMs) +
	// 		'ms]: <span style="display:none;" id="' +
	// 		eId +
	// 		'-hidden">' +
	// 		preview + "...</span>";
		
	// 	// Add Content
	// 	if (args.length === 1) {
	// 		nElement = document.createElement("span");
	// 		nElement.id = eId;
	// 		nElement.innerHTML = args[0].toString();
	// 		el.appendChild(nElement);
	// 	} else if (args.length > 1) {
	// 		nElement = document.createElement("ol");
	// 		nElement.id = eId;
	// 		nElement.style.margin = "0";
	// 		nElement.style.padding = "0";
	
	// 		for (let i = 0, len = args.length; i < len; i++) {
	// 			const item = document.createElement("li");
	// 			item.innerHTML = args[i].toString();
	// 			nElement.appendChild(item);
	// 		}

	// 		el.appendChild(nElement);
	// 	}
	
	// 	// Add Toggle Text if the combined string is long enough or there is more than one argument
	// 	if (allArgsString.length > 64 || args.length > 1) {
	// 		toggleElement.onclick = function (event) {
	// 			toggleVisible(event, eId, toggleElement);
	// 		};
	// 		toggleElement.innerHTML = "hide";
	// 		toggleElement.style.float = "right";
	// 		toggleElement.style.color = "black";
	// 		toggleElement.style.fontSize = "0.7em";
	// 		el.appendChild(toggleElement);
	// 	}
	
	// 	// Add line between entries
	// 	hrElement.style.clear = "both";
	// 	hrElement.style.margin = "0";
	// 	hrElement.style.padding = "0";
	// 	el.appendChild(hrElement);
	
	// 	// Add everything to the inline console
	// 	inlineConsole.appendChild(el);
	// 	if (el.clientHeight > consoleHeight) {
	// 		toggleVisible(null, eId, toggleElement);
	// 	}
	// 	inlineConsole.scrollTop = inlineConsole.scrollHeight;
	// };

	// console.log = function (...args) {
	// 	sendMsg("LOG", args);
	// 	oldLog.apply(console, arguments);
	// };

	// console.debug = function (...args) {
	// 	sendMsg("DEBUG", args, "#cccccc");
	// 	oldDebug.apply(console, arguments);
	// };

	// console.warn = function (...args) {
	// 	sendMsg("WARN", args, "#ff9900");
	// 	oldWarn.apply(console, arguments);
	// };

	// console.info = function (...args) {
	// 	sendMsg("INFO", args, "#0066ff");
	// 	oldInfo.apply(console, arguments);
	// };

	// console.error = function (...args) {
	// 	sendMsg("ERROR", args, "#ff0000");
	// 	oldError.apply(console, arguments);
	// };

	// for (let ei = 0, eLen = watchEvents.length; ei < eLen; ei++) {
	// 	document.addEventListener(
	// 		watchEvents[ei],
	// 		function (e) {
	// 			if (!isDescendant(consoleWrapper, e.target)) {
	// 				sendMsg(
	// 					"EVENT",
	// 					[
	// 						"An event type &quot;" +
	// 							e.type +
	// 							"&quot; was triggered by a &quot;" +
	// 							e.target.nodeName +
	// 							"&quot; node.",
	// 					],
	// 					"#cccccc"
	// 				);
	// 			}
	// 		},
	// 		true
	// 	);
	// }

	// /*document.addEventListener('error', function(e) {
    //     if(!isDescendant(consoleWrapper, e.target)) {
    //         sendMsg('EVENT-ERROR', ['An event type &quot;'+e.type+'&quot; was triggered by a &quot;'+e.target.nodeName+'&quot; node.'], '#ff0000');
    //     }

    // }, true);*/

	// window.onerror = function (...args) {
	// 	sendMsg("ERROR", args, "#ff0000");
	// };

	// window.onload = function () {
	// 	document.body.appendChild(consoleWrapper);
	// };

	// /**
	//  * Override and extend the default open/send methods for XMLHttpRequest so we can log this activity
	//  *
	//  * Based on code by Julien Couvreur (http://blog.monstuff.com/archives/cat_greasemonkey.html)
	//  * and included here with his gracious permission
	//  */
	// XMLHttpRequest.prototype.oldOpen = XMLHttpRequest.prototype.open;
	// XMLHttpRequest.prototype.oldSend = XMLHttpRequest.prototype.send;

	// XMLHttpRequest.prototype.getId = function () {
	// 	if (!this.threadId) {
	// 		this.threadId = ++XMLHttpRequestCount;
	// 	}
	// 	return ("00" + this.threadId).slice(-3);
	// };

	// XMLHttpRequest.prototype.open = function (
	// 	method,
	// 	url,
	// 	async,
	// 	user,
	// 	password
	// ) {
	// 	sendMsg(
	// 		"HTTP",
	// 		[
	// 			"Opening Connection #" + this.getId(),
	// 			{
	// 				method: method,
	// 				url: url,
	// 				async: async,
	// 				user: user,
	// 				password: password,
	// 			},
	// 		],
	// 		"#cccccc"
	// 	);
	// 	this.oldOpen(method, url, async, user, password);
	// };

	// XMLHttpRequest.prototype.send = function (a) {
	// 	this.addEventListener(
	// 		"load",
	// 		function () {
	// 			sendMsg(
	// 				"HTTP",
	// 				[
	// 					"Connection #" +
	// 						this.getId() +
	// 						" Completed Successfully: " +
	// 						this.status,
	// 				],
	// 				"#cccccc"
	// 			);
	// 		},
	// 		false
	// 	);
	// 	this.addEventListener(
	// 		"error",
	// 		function () {
	// 			sendMsg(
	// 				"HTTP-ERROR",
	// 				["Connection #" + this.getId() + " Failed: " + this.status],
	// 				"#ff0000"
	// 			);
	// 		},
	// 		false
	// 	);

	// 	this.oldSend(a);
	// };


// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();
