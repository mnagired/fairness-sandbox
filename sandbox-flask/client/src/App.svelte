<script>
  // import { Chart } from 'chart.js/dist/chart.js';
  import Bar from './Bar.svelte'

  let txt = '';
  let plot = '';
  let showPlot = false;

  let chartVar = 'sex';
  let chartKey = ['1','2','3','4','5','6'];
  let chartValue = [1,2,3,4,5,6];

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



        // data = d.text();
        // console.log(data);
        // const keys = Object.keys(data);
        // const values = Object.values(data);
        // console.log(keys);
        // console.log(values);
      // })
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
<input bind:value={chartVar}>
<button on:click={plotCounts}>Visualize</button>
<button on:click={() =>{showPlot = false}}>Hide</button>
<br/>
<button on:click={getBefore}>Before</button>
<button on:click={injectBias}>Inject Bias - Oversample</button>
<button on:click={injectBiasUnder}>Inject Bias - Undersample</button>

{#if showPlot}
  <Bar bind:data={chartValue} bind:labels={chartKey} bind:chartVar={chartVar}/>
{/if}

<p>{txt}</p>




<!-- <Bar data={chartValue} labels={chartKey}/> -->

<style>
	p {
	}
</style>
