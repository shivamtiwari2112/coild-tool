// Main script.js file for ASR Models Leaderboard

// Debug function to log information to console
function debug(message, data) {
    console.log(`DEBUG: ${message}`, data || '');
}

// Initialize page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    debug('DOM content loaded');
    
    try {
        // Add event listeners to the form elements
        const datasetSelect = document.getElementById('dataset');
        const languageSelect = document.getElementById('language');
        
        if (!datasetSelect || !languageSelect) {
            throw new Error('Select elements not found');
        }
        
        // Add change event listeners
        datasetSelect.addEventListener('change', function() {
            // Submit the form to reload the page with the new dataset
            document.getElementById('filter-form').submit();
        });
        
        languageSelect.addEventListener('change', function() {
            // Submit the form to reload the page with the new language
            document.getElementById('filter-form').submit();
        });
        
        // Check if we have chart data from the server
        initializeChart();
        
        debug('Event listeners added');
    } catch (error) {
        console.error('Initialization error:', error);
    }
});

// Function to initialize the chart with model data
function initializeChart() {
    try {
        debug('initializeChart called');
        
        // Get the chart canvas element
        const ctx = document.getElementById('wer-chart');
        if (!ctx) {
            debug('Chart canvas element not found');
            return;
        }
        
        // Get the models table
        const tableBody = document.getElementById('models-table-body');
        if (!tableBody) {
            debug('Models table body not found');
            return;
        }
        
        // Extract model data from the table
        const modelData = [];
        const rows = tableBody.querySelectorAll('tr');
        
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 4) { // Ensure we have enough cells
                const model = {
                    model: cells[1].textContent.trim(),
                    test_wer: parseFloat(cells[2].textContent),
                    year: parseInt(cells[3].textContent),
                    extra_training_data: cells[4].querySelector('.badge-extra-data') !== null
                };
                modelData.push(model);
            }
        });
        
        debug('Extracted model data', modelData);
        
        // If there's no data, don't attempt to create a chart
        if (modelData.length === 0) {
            debug('No data for chart');
            return;
        }
        
        // Process the data for visualization
        const modelsByYear = {};
        
        modelData.forEach(model => {
            // Ensure year is properly defined
            const year = model.year || 0;
            
            if (!modelsByYear[year]) {
                modelsByYear[year] = [];
            }
            
            modelsByYear[year].push({
                name: model.model || 'Unknown',
                wer: model.test_wer || 100
            });
        });
        
        // Prepare the dataset for the line chart
        const years = Object.keys(modelsByYear).sort();
        
        // If no years, don't create chart
        if (years.length === 0) {
            debug('No years data for chart');
            return;
        }
        
        // Create a line connecting the best model (lowest WER) for each year
        const bestModelData = years.map(year => {
            const bestModel = modelsByYear[year].reduce((best, current) => 
                current.wer < best.wer ? current : best, modelsByYear[year][0]);
            return {
                x: parseInt(year),
                y: bestModel.wer,
                model: bestModel.name
            };
        }).sort((a, b) => a.x - b.x);
        
        // Check if a chart already exists and destroy it
        if (window.wer_chart) {
            debug('Destroying existing chart');
            window.wer_chart.destroy();
        }
        
        debug('Creating new chart');
        window.wer_chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Best WER by Year',
                    data: bestModelData,
                    borderColor: '#2e86de',
                    backgroundColor: 'rgba(46, 134, 222, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#2e86de',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Year',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return value;
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'WER (%)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        suggestedMax: Math.max(...bestModelData.map(d => d.y)) * 1.2 || 10
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return [`Model: ${point.model}`, `WER: ${point.y}%`];
                            }
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
        
        // Add data points for each model
        debug('Adding individual model data points');
        const colorPalette = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
            '#1abc9c', '#d35400', '#34495e', '#7f8c8d', '#c0392b'
        ];
        
        modelData.forEach((model, index) => {
            // Skip if missing critical data
            if (!model.year || !model.test_wer) return;
            
            const colorIndex = index % colorPalette.length;
            
            window.wer_chart.data.datasets.push({
                label: model.model || 'Unknown',
                data: [{
                    x: model.year,
                    y: model.test_wer,
                    model: model.model || 'Unknown'
                }],
                borderColor: 'rgba(0, 0, 0, 0)',
                backgroundColor: colorPalette[colorIndex],
                pointBackgroundColor: colorPalette[colorIndex],
                pointRadius: 6,
                pointHoverRadius: 8,
                showLine: false
            });
        });
        
        window.wer_chart.update();
        debug('Chart updated successfully');
    } catch (error) {
        console.error('Chart error:', error);
    }
}

// Check if script loaded properly
debug('Script loaded');