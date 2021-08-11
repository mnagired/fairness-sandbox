const CHART_COLORS = {
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    blue: 'rgb(54, 162, 235)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(201, 203, 207)'
  };
  
const NAMED_COLORS = [
    CHART_COLORS.red,
    CHART_COLORS.orange,
    CHART_COLORS.yellow,
    CHART_COLORS.green,
    CHART_COLORS.blue,
    CHART_COLORS.purple,
    CHART_COLORS.grey,
];

export function generateDataForLinePlot(jsonData) {
    console.log(jsonData);
    const keys = Object.keys(jsonData);
    const values = Object.values(jsonData);
    console.log(keys);
    console.log(values);

    let labelLength =  keys.length - 1; // -1 because first key is the label
    let dataLength =  values[0].length;
    
    let datasets = []
    for (var i = 1; i <= labelLength; i++){
        let temp = {
            label: keys[i],
            data: values[i],
            borderColor: NAMED_COLORS[i],
            backgroundColor: NAMED_COLORS[i],
        }
        datasets.push(temp);
    }
    return [values[0], datasets];
}
