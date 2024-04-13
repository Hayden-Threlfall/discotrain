// Accessing webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    var video = document.getElementById('webcam');
    video.srcObject = stream;
    video.play();

    fetch('run_record.sh')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.text();
      })
      .then(data => {
        console.log(data);
      })
      .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
      });

  })
  .catch(function(err) {
    console.log("An error occurred: " + err);
  });

  document.getElementById("toggleWebcam").addEventListener("click", function() {
    var webcamContainer = document.getElementById("webcam-container");
    webcamContainer.classList.toggle("hidden");
});