$(document).ready(function(){
    $('.carousel').carousel();
  });

function toggleVideo() {
    const preview = document.querySelector('.preview');
    const video = document.querySelector('video');
    video.pause();
    preview.classList.toggle('active');
}