$(document).ready(function(){
    $('.carousel').carousel();
  });

function toggleVideo() {
    const preview = document.querySelector('.preview');
    const video = document.querySelector('video');
    video.pause();
    preview.classList.toggle('active');
}

function updateImage(index) {
    // Data arrays
    const images = ["LadyGaga.jpg", "callMeMaybe.jpg", "tiktok.jpg", "dynamite.jpg", "classic.jpg"];
    const titles = ["Just Dance", "Call Me Maybe", "Tik Tok", "Dynamite", "Classic"];
    const spans = [
        ["Lady Gaga", "2008", "Pop", "4:07"],
        ["Carly Rae Jepson", "2012", "Pop", "3:20"],
        ["Kesha", "2010", "Pop", "3:36"],
        ["Taio Cruz", "2010", "Pop", "3:23"],
        ["MKTO", "2013", "Pop", "2:54"]
    ];
    const descriptions = [
        "This is a song description 1.",
        "This is a song description 2.",
        "This is a song description 3.",
        "This is a song description 4.",
        "This is a song description 5."
    ];

    // Function to update content
    function updateContent(index) {
        const bannerImage = document.querySelector('.banner-image');
        const songTitle = document.querySelector('.song-title');
        const spansElements = document.querySelectorAll('.content.active h4 span');
        const description = document.querySelector('.content.active p');

        bannerImage.src = "images/" + images[index];
        songTitle.innerHTML = "<i>" + titles[index] + "</i>";

        // Update spans
        spansElements.forEach((spanElement, i) => {
            spanElement.textContent = spans[index][i];
        });

        description.textContent = descriptions[index];
    }

    // Initial call to update content
    updateContent(index);
};