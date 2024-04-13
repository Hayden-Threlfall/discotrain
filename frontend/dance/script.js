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

  function getAttachmentByTitle(title) {
    // Step 1: Fetch records based on title value
    fetch(`https://discotrain.kintone.com/k/v1/records.json?app=${1}&query=title="${title}"`, {
        method: 'GET',
        headers: {
            'X-Cybozu-API-Token': 'qYGRIRpfkjzRVFclHtaHeTQI1eusC7K5y7A0DKi4',
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        // Step 2: Extract record IDs
        const recordIds = data.records.map(record => record['$id']['value']);

        // Step 3: Fetch record details including attachments
        recordIds.forEach(recordId => {
            fetch(`https://discotrain.kintone.com/k/v1/record.json?app=${1}&id=${recordId}`, {
                method: 'GET',
                headers: {
                    'X-Cybozu-API-Token': 'qYGRIRpfkjzRVFclHtaHeTQI1eusC7K5y7A0DKi4',
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(recordData => {
                // Extract attachment URL(s) from record data
                const attachments = recordData.record['video'].value;

                // Use attachment URL(s) as needed
                console.log('Attachments:', attachments);
            })
            .catch(error => {
                console.error('Error fetching record details:', error);
            });
        });
    })
    .catch(error => {
        console.error('Error fetching records:', error);
    });
}

// Usage example: Call getAttachmentByTitle function when button is clicked
document.getElementById('getAttachmentButton').addEventListener('click', function() {
    getAttachmentByTitle('video1');
});
