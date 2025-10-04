document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const fileName = document.getElementById('fileName');
    const imagePreview = document.getElementById('imagePreview');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');

    // Handle file selection
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('Please select an image file');
            return;
        }

        // Hide previous results and errors
        results.classList.add('hidden');
        error.classList.add('hidden');
        loading.classList.remove('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data);
            } else {
                showError(data.error || 'Prediction failed');
            }
        } catch (err) {
            showError('Network error. Please try again.');
        } finally {
            loading.classList.add('hidden');
        }
    });

    function displayResults(data) {
        document.getElementById('predictedAge').textContent = `${data.predicted_age} months`;
        document.getElementById('predictedYears').textContent = `${data.predicted_years} years`;
        document.getElementById('confidence').textContent = data.confidence;
        results.classList.remove('hidden');
    }

    function showError(message) {
        error.textContent = message;
        error.classList.remove('hidden');
    }
});
