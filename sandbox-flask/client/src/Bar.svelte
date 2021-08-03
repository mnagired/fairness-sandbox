<script>
import {onMount, afterUpdate} from 'svelte';

export let data;
export let labels;
export let chartVar;
var ctx;
var myChart;

function createChart() {
    var ctx = document.getElementById('myChart').getContext('2d');
    // var ctx = document.getElementById('myChart').getContext('2d');
    // var myChart = new Chart(ctx, {
    console.log(data);
    
    if (myChart) {
        myChart.data.datasets[0].data = data;
        myChart.data.datasets[0].label = chartVar;
        myChart.data.labels = labels;
        myChart.update();
    } else {
        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: chartVar,
                    data: data,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
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
    max-width: 500px !important;
    max-height: 30px !important;
}

/*  */

h1{
    color:red;
}
</style>


<div class="chart-container">
    <canvas id="myChart" width="2" height="1"></canvas>
</div>

