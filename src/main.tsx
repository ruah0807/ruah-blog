import React from "react";
import ReactDOM from "react-dom/client";
import {
  RouterProvider,
  createHashRouter,
} from "react-router-dom";
import App from './App.tsx'
import './styles/global.css'; // global.css를 나중에 임포트
// import './App.css';

// import { Buffer } from 'buffer';

// // Example usage
// const buf = Buffer.from('Hello, world!', 'utf-8');
// console.log(buf.toString('hex'));

const router = createHashRouter([
  {
    path: "/ruah-blog/",
    element: <App />,
    loader: () => <div>로딩중...</div>,
    ErrorBoundary: () => <div>404 NOT FOUND</div>,
  }
]);


ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
      <RouterProvider router={router} />
  </React.StrictMode>
);
