import React from 'react';
import './Header.css';
import logo from '../../public/ruah_blog.svg';
import { useNavigate } from 'react-router-dom';

interface HeaderProps {
  toggleSidebar: () => void;
}

const Header: React.FC<HeaderProps> = ({ toggleSidebar }) => {
  const navigate = useNavigate();

  const handleTitleClick = () => {
    window.location.href = '/ruah-blog/'; // Navigate to the main page
  };

  const handleMenuClick = () => {
    navigate('/menu'); // URL 변경
    toggleSidebar(); // 사이드바 토글
  };

  return (
    <header>
      <div className='header-title' onClick={handleTitleClick} style={{ cursor: 'pointer' }}>
        <div className='header-back-container'>
          <button className='header-back' onClick={() => navigate(-1)}>❮ </button>
          <img className='header-logo' draggable="false" src={logo} alt="logo" />
        </div>
      </div>
      <nav>
        <ul>
          <li>About Me</li>
          <li className='hamburger' onClick={handleMenuClick}>
            목록보기
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 