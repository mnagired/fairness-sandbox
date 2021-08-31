<script>
import {onMount, afterUpdate} from 'svelte';

// export let data;
// export let labels;
// export let chartVar;
export let labels;
export let dataset;

var ctx;
var myChart;


function createChart() {
    var ctx = document.getElementById('lineChart').getContext('2d');
    // var myChart = new Chart(ctx, {
    // console.log(data);
    
    if (myChart) {
        myChart.data.labels = labels;
        myChart.data.datasets = dataset;
        // myChart.data.datasets[0].label = chartVar;
        // myChart.data.labels = labels;
        myChart.update();
    } else {
        myChart = new Chart(ctx, {
            type: 'line',
            data: {
            labels: labels,
            datasets: dataset
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }
}

// onMount(createChart);
// afterUpdate(createChart);
afterUpdate(() =>{
    createChart();
});


</script>

<style>
.chart-container {
    position: relative;
    max-width: 800px !important;
    max-height: 200px !important;
}
</style>


<div class="chart-container">
    <canvas id="lineChart" width="2" height="1"></canvas>
</div>

