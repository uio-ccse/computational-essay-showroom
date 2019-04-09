// Add modal to Jupyter's png figures.
// 20190409 Alessandro Marin
const classSelector = "output_png";
const getElements = () => document.getElementsByClassName(classSelector);

const callback = event => {
  var elem = event.target;
  if (event.target.nodeName === "IMG") {
    elem = event.target.parentElement;
    console.log("elem.nodeName:", elem.nodeName);
  }
  var image = elem.firstElementChild;
  // Append a modal opening on click
  modal = document.createElement("div");
  modal.className = "modal";
  span = document.createElement("span");
  span.classList.add("close");
  span.innerHTML = "&times";
  img = document.createElement("img");
  img.classList.add("modal-content");
  img.src = image.src;
  modal.appendChild(span);
  modal.appendChild(img);
  // Append modal to the js-textbook div so that it takes the whole screen
  document.getElementById("js-textbook").appendChild(modal);  
  // When the user clicks on <span> (x) or the div, remove the modal
  span.onclick = () => {
    modal.remove();
  };
  // Exit from modal if the users freaks out and randomly double clicks
  modal.ondblclick = () => {
    modal.remove();
  };
};

const alessandroHandler = () => {
  let elements = getElements();
  for (var i = 0; i < elements.length; i++) {
    elements[i].addEventListener("click", callback);
  }
};

initFunction(alessandroHandler);
