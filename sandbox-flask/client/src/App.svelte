<script>
  let txt = '';
  let plot = '';
  let plotVar = 'sex'

  function getGender() {
    fetch(`./getGender/${plotVar}`)
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
<button on:click={getGender}>Visualize</button>
<br/>
<button on:click={getBefore}>Before</button>
<button on:click={injectBias}>Inject Bias - Oversample</button>
<button on:click={injectBiasUnder}>Inject Bias - Undersample</button>
<p>{txt}</p>
<img src={plot}>


