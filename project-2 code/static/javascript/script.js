function displayInputs(){
    var element = document.getElementById("form-below");
    element.scrollIntoView({behavior: "smooth"});
}

document.getElementById('algo').addEventListener('change', function(){
    if (document.getElementById('algo').value === 'Regression')
    {
        document.getElementById('regression-metric').style.display = 'inline';
        document.getElementById('classification-metric').style.display = 'none';
    }
    else
    {
        document.getElementById('regression-metric').style.display = 'none';
        document.getElementById('classification-metric').style.display = 'inline';
    }
})

function showSpinner()
{
    document.getElementsByClassName('spinner-div')[0].style.display = 'inherit'
}


document.getElementById('dataset').onchange = function() {
    file_input = document.getElementById('dataset').files[0]
    const reader = new FileReader();
    reader.readAsText(file_input);
    reader.onload = function (e) {
        headings = getHeadings(e.target.result)
        console.log(headings)
        target_tag = document.getElementById('target')
        redundant_tag = document.getElementById('redundant-features')
        //target column
        for(let i=0; i<headings.length; i++)
        {
            option = document.createElement('option')
            option.innerText = headings[i]
            option.setAttribute('value', headings[i])
            target_tag.appendChild(option)
        }
        //redundant columns
        for(let i=0; i<headings.length; i++)
        {
            option = document.createElement('option')
            option.innerText = headings[i]
            option.setAttribute('value', headings[i])
            redundant_tag.appendChild(option)
        }
        redundant_tag.style.height = '80px'
    }
    
}

function getHeadings(str) {

    headings = []

    headings = str.slice(0, str.indexOf("\r\n")).split(',');
    
    return headings
  }