<script>
  // import { Chart } from 'chart.js/dist/chart.js';
  import Bar from './Bar.svelte'
  import Line from './Line.svelte'
  import {generateDataForLinePlot} from './util/utils.js'

  let txt = '';
  let plot = '';
  let showPlot = false;
  let showLine = false;

  let chartVar = 'sex';
  let chartKey = ['1','2','3','4','5','6'];
  let chartValue = [1,2,3,4,5,6];
  let lineLabels = [];
  let lineDataset = [];

  function plotCounts() {
    fetch(`./plotCounts/${chartVar}`)
      .then(d => d.text())
      .then(d =>{
        let data = JSON.parse(d);

        const keys = Object.keys(data);
        const values = Object.values(data);

        chartValue = values;
        chartKey = keys;
        console.log(chartKey);
        console.log(chartValue);
        showPlot = true;
      })
  }

  function getBefore() {
    fetch("./getBefore")
      .then(d => d.text())
      .then(d => (txt = d));
  }

  function injectBias() {
    fetch("./injectBias")
      .then(d => d.text())
      .then(d => (txt = d));
  }

  function trainModel() {
    fetch("./trainModel")
      .then(d => d.text())
      .then(d =>{
        txt = "";
        console.log(d);
        let data = JSON.parse(d);
        // let data = d;

        const keys = Object.keys(data);
        const values = Object.values(data);

        chartValue = values;
        chartKey = keys;
        chartVar = "Accuracy";
        console.log(chartKey);
        console.log(chartValue);
        showPlot = true;
    })
  }

  function fairnessIntervention() {
    showPlot = false;
    txt = "loading";
    fetch("./fairnessIntervention")
      .then(d => d.text())
      .then(d => (txt = d));
  }

  function fairnessTradeoff() {
    showPlot = false;
    txt = "Processing...";    
    fetch("./fairnessTradeoff")
      .then(d => d.text())
      .then(d =>{
        txt = "";
        console.log(d);
        let data = JSON.parse(d);
        let tmp = generateDataForLinePlot(data);
        lineLabels = tmp[0];
        lineDataset = tmp[1];
        console.log(lineDataset);
        showLine = true;
    });
  }
</script>



<svelte:head>
	<title>Fairness Sandbox</title>
	<meta name="robots" content="noindex nofollow" />
	<html lang="en" />
</svelte:head>

<h1>Fairness Sandbox</h1>
<input bind:value={chartVar}>
<button on:click={plotCounts}>Visualize</button>
<button on:click={() =>{showPlot = false}}>Hide</button>
<br/>
<!-- <button on:click={getBefore}>Before</button> -->
<button on:click={injectBias}>Inject Bias</button>
<button on:click={trainModel}>Train Model</button>
<button on:click={fairnessIntervention}>Fairness Intervention</button>
<button on:click={fairnessTradeoff}>Fairness Trade-off</button>

{#if showPlot}
  <Bar bind:data={chartValue} bind:labels={chartKey} bind:chartVar={chartVar}/>
{/if}

{#if showLine}
  <Line bind:labels={lineLabels} bind:dataset={lineDataset}/>
{/if}

<p>{txt}</p>




<!-- <Bar data={chartValue} labels={chartKey}/> -->

<style>
	p {
	}
</style>
