// College Recommender - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Form validation
    const recommendationForm = document.querySelector('form');
    if (recommendationForm) {
        recommendationForm.addEventListener('submit', function(event) {
            const userInput = document.getElementById('user_input');
            
            // Validate user input
            if (userInput.value.trim().length < 20) {
                event.preventDefault();
                
                // Create alert
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-warning alert-dismissible fade show mt-3';
                alertDiv.setAttribute('role', 'alert');
                alertDiv.innerHTML = `
                    Please provide more details about your preferences (at least 20 characters) for better recommendations.
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                
                // Insert alert before the form
                recommendationForm.parentNode.insertBefore(alertDiv, recommendationForm);
                
                // Focus on input
                userInput.focus();
            }
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70, // Account for fixed navbar
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animation for match score circles
    const matchScoreCircles = document.querySelectorAll('.match-score-circle');
    if (matchScoreCircles.length > 0) {
        // Add animation when elements are in viewport
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Add animation class
                    entry.target.classList.add('animated');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.2
        });
        
        matchScoreCircles.forEach(circle => {
            observer.observe(circle);
        });
    }
    
    // Autocomplete for search (if implemented)
    const searchInput = document.getElementById('college-search');
    if (searchInput) {
        // Fetch colleges list from the API
        fetch('/api/colleges')
            .then(response => response.json())
            .then(data => {
                // Initialize autocomplete
                const collegeNames = data.colleges;
                
                // Simple autocomplete implementation
                searchInput.addEventListener('input', function() {
                    const inputValue = this.value.toLowerCase();
                    
                    // Get datalist
                    let datalist = document.getElementById('college-list');
                    
                    // Create datalist if it doesn't exist
                    if (!datalist) {
                        datalist = document.createElement('datalist');
                        datalist.id = 'college-list';
                        this.parentNode.appendChild(datalist);
                        
                        // Connect input to datalist
                        this.setAttribute('list', 'college-list');
                    }
                    
                    // Clear existing options
                    datalist.innerHTML = '';
                    
                    // Add matching colleges
                    if (inputValue.length > 2) {
                        const matches = collegeNames.filter(name => 
                            name.toLowerCase().includes(inputValue)
                        ).slice(0, 5);
                        
                        matches.forEach(name => {
                            const option = document.createElement('option');
                            option.value = name;
                            datalist.appendChild(option);
                        });
                    }
                });
            })
            .catch(error => console.error('Error fetching colleges:', error));
    }
});
