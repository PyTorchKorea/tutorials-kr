document.addEventListener('DOMContentLoaded', function() {
    const dropdownButtons = document.querySelectorAll('[data-toggle]');

    dropdownButtons.forEach(function(button) {
        const dropdown = button.closest('.learn-dropdown, .docs-dropdown, .community-dropdown');
        if (dropdown) {
            dropdown.addEventListener('mouseenter', function() {
                const menu = this.querySelector('.dropdown-menu');
                if (menu) {
                    menu.style.display = 'block';
                }
            });

            dropdown.addEventListener('mouseleave', function() {
                const menu = this.querySelector('.dropdown-menu');
                if (menu) {
                    menu.style.display = 'none';
                }
            });
        }
    });

    const mobileMenuButton = document.querySelector('[data-behavior="open-mobile-menu"]');
    const mobileMenu = document.querySelector('.mobile-main-menu');
    const closeMenuButton = document.querySelector('[data-behavior="close-mobile-menu"]');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function(e) {
            e.preventDefault();
            mobileMenu.classList.add('open');
        });
    }

    if (closeMenuButton && mobileMenu) {
        closeMenuButton.addEventListener('click', function(e) {
            e.preventDefault();
            mobileMenu.classList.remove('open');
        });
    }
});