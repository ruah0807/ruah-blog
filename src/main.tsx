import App from './App.tsx'
import './styles/global.css'; 
import React from 'react';
import ReactDOM from 'react-dom/client';
import{ RouterProvider, createHashRouter} from "react-router-dom"

const router = createHashRouter([
  {
    path: "/ruah-blog",
    element: <App />,
  },
]);

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)