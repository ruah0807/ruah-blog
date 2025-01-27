import { useState } from "react";
import Header from "../Header/Header";
import { Outlet } from "react-router";
import PostList from "../../pages/PostList";
import styles from "./RootLayout.module.scss";

const RootLayout = () => {
  const [showSidebar, setShowSidebar] = useState<boolean>(false);

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar);
  };

  return (
    <>
      <Header toggleSidebar={toggleSidebar} />
      <div
        className={`${styles.container} ${
          showSidebar ? `${styles.show_sidebar}` : ""
        }`}
      >
        <div className={styles.content_wrapper}>
          <div className={styles.post_list_container}>
            <PostList />
          </div>
          <div className={styles.post_container}>
            <Outlet />
          </div>
        </div>
      </div>
    </>
  );
};

export default RootLayout;
