// custom javascript


$(function() {
  console.log('jquery is working!')
  createGraph();
});

function createGraph() {

  // main config
  var width = 500; // chart width
  var height = 500; // chart height
  var format = d3.format(",d");  // convert value to integer
  var color = d3.scale.category20b();  // create ordial scale with 20 colors
  var sizeOfRadius = d3.scale.pow().domain([-100,100]).range([-50,50]);  // https://github.com/mbostock/d3/wiki/Quantitative-Scales#pow
  var xPosition = d3.scale.linear().domain([0,1]).range([0,width])
  var yPosition = d3.scale.linear().domain([0,1]).range([0,height])

  // bubble config
  //var bubble = d3.layout.pack()
  //  .sort(null)  // disable sorting, use DOM tree traversal
  //  .size([width, height])  // chart layout size
  // .padding(1)  // padding between circles
  //  .radius(function(d) { return 20 + (sizeOfRadius(d) * 60); });  // radius for each circle

  // svg config
  var svg = d3.select("#chart").append("svg") // append to DOM
    .attr("width", width)
    .attr("height", height)
    .attr("class", "bubble");

  // tooltip config
  var tooltip = d3.select("body")
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

  // request the data
  d3.json("/data", function(error, quotes) {
    console.log(quotes)
    var node = svg.selectAll('.node')
      .data(quotes.children)
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', function(d) { return 'translate(' + xPosition(d.guesses.bug)+ ',' + xPosition(d.guesses.feature) + ')'});

  node.append("circle")
    .attr("r", 10)
    .style('fill', function(d) { return color(d.symbol); })
    .on("mouseover", function(d) {
      tooltip.text(d.title);
      tooltip.style("visibility", "visible");
    })
    .on("mousemove", function() {
      return tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
    })
   .on("mouseout", function(){return tooltip.style("visibility", "hidden");});

  //  node.append('text')
  //    .attr("dy", ".3em")
  //    .style('text-anchor', 'middle')
  //    .text(function(d) { return d.symbol; });

  });

}