<script>
  // import { Chart } from 'chart.js/dist/chart.js';
  import Bar from './Bar.svelte'
  import Line from './Line.svelte'
  import {generateDataForLinePlot} from './util/utils.js'

  let txt = '';
  let plot = '';
  let showPlot = false;
  let showLine = false;

  let chartVar = 'sens_feat';
  let chartKey = ['1','2','3','4','5','6'];
  let chartValue = [1,2,3,4,5,6];
  let lineLabels = [];
  let lineDataset = [];

  let selectedbias = "representation";

  function plotCounts() {
    showLine = false;
    fetch(`./plotCounts/${chartVar}`)
      .then(d => d.text())
      .then(d =>{
        let data = JSON.parse(d);

        const keys = Object.keys(data);
        const values = Object.values(data);

        chartValue = values;
        chartKey = keys;
        showPlot = true;
      })
  }


  function injectBias() {
    showLine = false;
    console.log(selectedbias);
    fetch(`./injectBias/${selectedbias}`)
      .then(d => d.text())
      .then(d => (txt = d));
  }

  function trainModel() {
    showLine = false;
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

<div class="flex items-center">
  <div class="" >
    <label for="dataset">Dataset:</label>
  </div>
  <div class="w-25 mr2" >
    <select name="dataset" id="dataset" disabled>
      <option value="synthetic">Synthetic</option>
    </select>
  </div>
</div>

<br/>
<div class="flex items-center">
  <p>Feature: </p>
  <div class="w-25 mr2" >
    <input bind:value={chartVar}>
  </div>
  <div class="w-25 mr2" >
    <button on:click={plotCounts}>Visualize</button>
    <button on:click={() =>{showPlot = false}}>Hide</button>
  </div>
</div>

<br/>

<div class="flex items-center">
  <p>Select Bias: </p>
  <div class="w-25 mr2" >
    <select name="bias" id="bias" bind:value={selectedbias}>
      <option value="representation">Representation</option>
      <option value="measurement">Measurement</option>
      <option value="omitted_variable">Omitted Variable</option>
      <option value="label_noise">Label Noise</option>
      <option value="over_sampling">Over-Sampling</option>
      <option value="under_sampling">Under-Sampling</option>
    </select>
  </div>
  <div class="w-25 mr2" >
    <button on:click={injectBias}>Inject Bias</button>
  </div>
</div>

<br/>

<div class="flex items-center">
  <p>Select Model: </p>
  <div class="w-25 mr2" >
    <select name="model" id="model" disabled>
      <option value="logistic_regression">Logistic Regresion</option>
    </select>
  </div>
  <div class="w-25 mr2" >
    <button on:click={trainModel}>Train Model</button>
  </div>
</div>

<br/>

<div class="flex items-center">
  <p>Select Fairness Intervention: </p>
  <div class="w-25 mr2" >
    <select name="fairnessIntervention" id="fairnessIntervention" disabled>
      <option value="correlation">Correlation Remover</option>
    </select>
  </div>
  <div class="w-25 mr2" >
    <button on:click={fairnessIntervention}>Add Fairness Intervention</button>
  </div>
</div>

<br/>

<div class="flex items-center">
  <p>Select Metrics: </p>
  <div class="w-25 mr2" >
    <select name="metrics" id="metrics" disabled>
      <option value="accuracy">Accuracy</option>
    </select>
  </div>
  <div class="w-25 mr2" >
    <button on:click={fairnessTradeoff}>Fairness Trade-off</button>
  </div>
</div>

<p>{txt}</p>


{#if showPlot}
  <Bar bind:data={chartValue} bind:labels={chartKey} bind:chartVar={chartVar}/>
{/if}

{#if showLine}
  <Line bind:labels={lineLabels} bind:dataset={lineDataset}/>
{/if}





<!-- <Bar data={chartValue} labels={chartKey}/> -->

<style>
	p {
	}
</style>
