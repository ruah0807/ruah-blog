import { useState } from "react";
import Header from "../Header/Header";
import { Outlet } from "react-router";
import PostList from "../../pages/PostList";

const RootLayout = () => {
  const [showSidebar, setShowSidebar] = useState<boolean>(false);

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar);
  };

  return (
    <>
      <Header toggleSidebar={toggleSidebar} />
      <div className={`container ${showSidebar ? "show-sidebar" : ""}`}>
        <div className="content-wrapper">
          <div className="post-list-container">
            <PostList />
          </div>
          <div className="post-container"></div>
          <Outlet />
        </div>
      </div>
    </>
  );
};

export default RootLayout;
