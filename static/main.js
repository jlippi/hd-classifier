// custom javascript


$(function() {
  console.log('jquery is working!')
  createGraph();
});

var data;
var details_url; // this is necessary so that the details page can be refreshed

var tooltip;
var curr_cat = 'feature'

var labels = {};
var label_colors = d3.scale.category10();

function refreshData() {
  $.get("/data").done(function (d) {
      data = d;
      for (i in data.children) {
        labels = {};
        for (key in data.children[i].guesses) {
          labels[key] = true;
        }
      }
      drawButtons();
      drawGraph();
  });
}

function drawButtons() {
  d3.selectAll(".labelText").remove();
  var buttons = d3.select(".text_div");


  for (label in labels) {
    buttons
      .append("div")
      .datum(label)
      .on("click", function(l) {
        curr_cat = l; 
        drawGraph(); })
      .text(label)
      .attr("class","labelText")
      .style("background-color", label_colors)
    }
}

function drawGraph() {
  var tickets = data
  var width = d3.select(".container").style('width')
  var height = width/1.618
  var xPosition = d3.scale.linear().domain([tickets.children.length,0]).range([0,width])
  var yPosition = d3.scale.linear().domain([0,1]).range([height,0])
  var radiusScaler = d3.scale.pow().domain([1,5]).range([10,35])
  var color = d3.scale.category20b();  // create ordial scale with 20 colors

  d3.select(".bubble").remove();

  var svg = d3.select("#chart").append("svg") // append to DOM
    .attr("width", width)
    .attr("height", height)
    .attr("class", "bubble");

  var node = svg.selectAll('.node')
      .data(tickets.children)
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', function(d,i) { 
        if (curr_cat in d.guesses) {
          return 'translate(' + xPosition(i) + ',' + yPosition(d.guesses[curr_cat]) + ')'
        }
        return ''}
          );

  node.append("circle")
      .attr("r", function(d) { return radiusScaler(d.priority); })
      .style("visibility", function(d) {
        if ( d.guesses[curr_cat] === undefined ) {
          return "hidden";
        }
        return "visible";
      })
      .style("fill", function(d) {
        var best_guess_label;
        var best_guess = 0;
        for (guess in d.guesses) {
          if ( d.guesses[guess] > best_guess ) {
            best_guess = d.guesses[guess];
            best_guess_label = guess;
          }
        }
        return label_colors(best_guess_label);
      })
      .on("mouseover", function(d) {
          tooltip.text(d.title);
          tooltip.style("visibility", "visible");
      })
      .on("mousemove", function() {
          return tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
      })
      .on("click", function(d) { details_url = d.url; showDetails(); })
      .on("mouseout", function(){return tooltip.style("visibility", "hidden");})
}

function showDetails() {
   url = details_url;
   $.get(url).done( function(d)  {
     $('.ticket_detail').html(d); });
}

function createGraph() {

  // main config

  var format = d3.format(",d");  // convert value to integer

  tooltip = d3.select("body")
    .append("div")
    .style("position", "absolute")
    .style("z-index", "10")
   .style("visibility", "hidden")
    .style("color", "white")
    .style("padding", "8px")
    .style("background-color", "rgba(0, 0, 0, 0.75)")
    .style("border-radius", "6px")
   .style("font", "12px sans-serif")
    .text("tooltip");

  refreshData();

  d3.select('.container').on("resize",drawGraph);

}