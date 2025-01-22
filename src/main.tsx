import React from 'react';
import ReactDOM from 'react-dom';
import App from './App.tsx'
import './styles/global.css'; // global.css를 나중에 임포트
// import './App.css';

import { Buffer } from 'buffer';

// Example usage
const buf = Buffer.from('Hello, world!', 'utf-8');
console.log(buf.toString('hex'));

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
