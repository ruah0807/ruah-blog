import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/global.scss"; // global.css를 나중에 임포트
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from "react-router";
import RootLayout from "./components/feature/RootLayout.tsx";
import MDFile from "./pages/MDFile.tsx";
import "./index.scss";
// window.addEventListener("contextmenu", (e) => e.preventDefault()); // 페이지 전체 우클릭 방지

const router = createBrowserRouter(
  createRoutesFromElements(
    // <Route path="/" element={<RootLayout />} errorElement={<>에러페이지</>}>
    <Route path="/" element={<RootLayout />}>
      <Route
        index={true}
        element={<div className="main-post">롸그에 오신것을 환영합니다.</div>}
      />
      <Route path=":category/:subtitle" element={<MDFile />} />
    </Route>
  ),
  {
    basename: "/ruah-blog/",
  }
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
