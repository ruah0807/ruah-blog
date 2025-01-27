import React from "react";
import logo from "../../../public/ruah_blog.svg";
import { useNavigate } from "react-router";
import styles from "./Header.module.scss";

interface HeaderProps {
  toggleSidebar: () => void;
}

const Header: React.FC<HeaderProps> = ({ toggleSidebar }) => {
  const navigate = useNavigate();

  const handleTitleClick = () => {
    navigate("/"); // Navigate to the main page
  };

  const handleMenuClick = () => {
    toggleSidebar(); // 사이드바 토글
  };

  return (
    <header>
      <div className={styles.header_wrapper}>
        <div className={styles.header_back_container}>
          <button className={styles.header_back} onClick={() => navigate(-1)}>
            ❮{" "}
          </button>
          <div className={styles.header_title} onClick={handleTitleClick}>
            <img
              className={styles.header_logo}
              draggable="false"
              src={logo}
              alt="logo"
            />
          </div>
        </div>
        <nav>
          <ul>
            {/* <li className='header-nav-item'>About Me</li> */}
            <li className={styles.header_nav_menu} onClick={handleMenuClick}>
              목록보기
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
