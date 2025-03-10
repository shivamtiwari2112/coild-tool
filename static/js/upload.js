
document.addEventListener('DOMContentLoaded', function() {
    // Show success modal if redirected with success parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('success') === 'true') {
        const successModal = new bootstrap.Modal(document.getElementById('successModal'));
        successModal.show();
        
        // Remove the query parameter from URL
        window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    // Handle dataset change to load languages
    const datasetSelect = document.getElementById('dataset');
    const languageSelect = document.getElementById('language');
    
    if (datasetSelect && languageSelect) {
        datasetSelect.addEventListener('change', function() {
            const dataset = this.value;
            
            // Clear current options
            languageSelect.innerHTML = '';
            
            if (dataset) {
                // Fetch languages for the selected dataset
                fetch(`/api/languages?dataset=${dataset}`)
                    .then(response => response.json())
                    .then(languages => {
                        languages.forEach(language => {
                            const option = document.createElement('option');
                            option.value = language;
                            option.textContent = language;
                            languageSelect.appendChild(option);
                        });
                    })
                    .catch(error => {
                        console.error('Error fetching languages:', error);
                    });
            }
        });
        
        // Trigger change event to load languages for the default dataset
        if (datasetSelect.value) {
            datasetSelect.dispatchEvent(new Event('change'));
        }
    }
    
    // Validate file uploads
    const form = document.getElementById('model-upload-form');
    
    if (form) {
        form.addEventListener('submit', function(event) {
            let isValid = true;
            
            // Check if model name is provided
            const modelName = document.getElementById('model-name').value.trim();
            if (!modelName) {
                isValid = false;
                showError('model-name', 'Model name is required');
            }
            
            // Check if model weights file is provided
            const modelWeights = document.getElementById('model-weights').files[0];
            if (!modelWeights) {
                isValid = false;
                showError('model-weights', 'Model weights file is required');
            }
            
            // Check if requirements.txt is provided
            const requirementsFile = document.getElementById('requirements-file').files[0];
            if (!requirementsFile) {
                isValid = false;
                showError('requirements-file', 'Requirements file is required');
            } else if (requirementsFile.name !== 'requirements.txt') {
                isValid = false;
                showError('requirements-file', 'File must be named requirements.txt');
            }
            
            // Check if model_inference.py is provided
            const inferenceFile = document.getElementById('inference-file').files[0];
            if (!inferenceFile) {
                isValid = false;
                showError('inference-file', 'Inference file is required');
            } else if (inferenceFile.name !== 'model_inference.py') {
                isValid = false;
                showError('inference-file', 'File must be named model_inference.py');
            }
            
            if (!isValid) {
                event.preventDefault();
            }
        });
    }
    
    // Function to show error for a field
    function showError(fieldId, message) {
        const field = document.getElementById(fieldId);
        field.classList.add('is-invalid');
        
        // Check if error message element already exists
        let errorElement = field.nextElementSibling;
        if (!errorElement || !errorElement.classList.contains('invalid-feedback')) {
            errorElement = document.createElement('div');
            errorElement.classList.add('invalid-feedback');
            field.parentNode.insertBefore(errorElement, field.nextSibling);
        }
        
        errorElement.textContent = message;
    }
});


