import { useState } from "react";
import "./styles.css"
import axios from "axios";

interface Response {
  classified_as: string,
  status: string,
  probability?: string
}

function App() {

  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<Response | null>(null);
  const handleButton = async () => {
    if(file != undefined) {
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await axios({
          method: "post",
          url: "http://127.0.0.1:5000/",
          data: formData,
          headers: { "Content-Type": "multipart/form-data" },
        });
        setResponse(response.data);
      } catch(error) {
        console.log(error)
      }
    }
  }

  const handleInputChange = (event: any) => {
    setFile(event.target.files[0]);
  }

  return (
    <>
      <div className="page-wrapper">
        <h1 className="title">What fruit is it?</h1>
        <h3 className="message">Add an image to find out.</h3>
        <input className="input-file" type="file" accept=".jpg" onChange={handleInputChange}></input>
        <button className="check-button" onClick={handleButton}>Check!</button>
        {
          response && response.probability &&
            <>
              <p className="response">This fruit is a {response.classified_as} (with a probability of {response.probability})</p>
            </>
        }
      </div>
    </>
  )
}

export default App
