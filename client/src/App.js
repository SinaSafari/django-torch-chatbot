import React, { useState } from 'react';
import './App.css';

function App() {

  const [botMsg, setBotMsg] = useState('')
  const [userMsg, setUserMsg] = useState('')


  const submitHandler = async (e) => {
    e.preventDefault()
    const res = await fetch('http://localhost:8000/api/chat', {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ "sentence": userMsg })
    })

    const data = await res.json()
    setBotMsg(data.message)
  }


  return (
    <div className="pure-g cform">
      <div className="pure-u-1-3">
        <form className="pure-form pure-form-stacked" onSubmit={(e) => submitHandler(e)}>
          <label htmlFor="inputtext">Chat to the bot</label>
          <input
            className="pure-input-rounded cinput"
            type="text"
            name="inputtext"
            placeholder="please type something"
            value={userMsg}
            onChange={(e) => setUserMsg(e.target.value)}
          />
          <br />
          <button className="pure-button" type="submit">Send</button>
        </form>
        <hr />
        <br />
        {botMsg && (
          <>
            <p style={{ fontSize: "20px", display: 'block' }}>Bot:</p>
            <div className="pure-g">
              <form className="pure-form pure-u-1-3">
                <input className="cinput" type="text" value={botMsg} readOnly />
              </form>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
