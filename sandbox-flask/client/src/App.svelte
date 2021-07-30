<script>
  import { Chart } from 'chart.js/dist/chart.js';
  import Bar from './Bar.svelte'

  let txt = '';
  let plot = '';
  let plotVar = 'sex';
  let data = [1,2,3,4,5,6];

  function plotCounts() {
    fetch(`./plotCounts/${plotVar}`)
      .then(d => d.text())
      .then(d => (plot = d));
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
  
  function injectBiasUnder() {
    fetch("./injectBiasUnder")
      .then(d => d.text())
      .then(d => (txt = d));
  }
</script>

<svelte:head>
	<title>Fairness Sandbox</title>
	<meta name="robots" content="noindex nofollow" />
	<html lang="en" />
</svelte:head>

<h1>Fairness Sandbox</h1>
<input bind:value={plotVar}>
<button on:click={plotCounts}>Visualize</button>
<br/>
<button on:click={getBefore}>Before</button>
<button on:click={injectBias}>Inject Bias - Oversample</button>
<button on:click={injectBiasUnder}>Inject Bias - Undersample</button>
<p>{txt}</p>
<img src={plot}>

<h1>hello</h1>


<input type="number" bind:value={data[0]}>
<input type="number" bind:value={data[1]}>
<input type="number" bind:value={data[2]}>
<input type="number" bind:value={data[3]}>
<input type="number" bind:value={data[4]}>
<input type="number" bind:value={data[5]}>

<Bar data={data} labels={['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange']}/>


